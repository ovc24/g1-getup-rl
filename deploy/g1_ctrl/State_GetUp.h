// State_GetUp: g1_ctrl state that runs the get-up policy until the robot stands still in
// the walking pose, then hands over to the walking state.
//
// Like State_RLBase (same deploy.yaml format, same ONNX runner, 50 Hz policy thread), with
// these differences:
//   - it first waits, motors limp, until the robot lies still for settle_s (at most
//     max_settle_s): the policy was trained from robots at rest, and the buttons can be
//     pressed while the robot is still rolling;
//   - no "bad orientation -> Passive" check: the robot starts lying on the ground;
//   - the gains go to the motors of the policy joints (joint_ids_map);
//   - the first policy step runs in enter(), so the first targets written are the policy's;
//   - it ends by itself: "standing" for hold_s (tilt < max_tilt, pelvis angular speed <
//     max_gyro, every leg joint within max_leg_dev of the walking pose) once min_time_s
//     have passed -> on_success (default Velocity); not standing after max_time_s ->
//     on_timeout (default Passive).
//
// config.yaml (FSM section), e.g.:
//   GetUp:
//     transitions:
//       Passive: LT + B.on_pressed
//     policy_dir: config/policy/getup
//     settle_s: 0.5         # still this long before the policy starts (s)
//     max_settle_s: 3.0     # start anyway after this long (s)
//     min_time_s: 2.0       # from the policy start, like max_time_s
//     hold_s: 0.5
//     max_time_s: 10.0
//     max_tilt: 0.35        # rad
//     max_gyro: 0.8         # rad/s
//     max_leg_dev: 0.5      # rad
//     on_success: Velocity
//     on_timeout: Fallen
#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "unitree_articulation.h"

class State_GetUp : public FSMState
{
public:
    State_GetUp(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

        env_ = std::make_unique<isaaclab::ManagerBasedRLEnv>(
            YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
            std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
        );
        env_->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

        kd_ = (cfg["kd"] ? cfg["kd"] : param::config["FSM"]["Passive"]["kd"]).as<std::vector<float>>();
        settle_s_ = read(cfg, "settle_s", 0.5f);
        max_settle_s_ = read(cfg, "max_settle_s", 3.0f);
        still_gyro_ = read(cfg, "still_gyro", 0.5f);
        still_joint_vel_ = read(cfg, "still_joint_vel", 1.0f);
        min_time_s_ = read(cfg, "min_time_s", 2.0f);
        hold_s_ = read(cfg, "hold_s", 0.5f);
        max_time_s_ = read(cfg, "max_time_s", 10.0f);
        max_tilt_ = read(cfg, "max_tilt", 0.35f);
        max_gyro_ = read(cfg, "max_gyro", 0.8f);
        max_leg_dev_ = read(cfg, "max_leg_dev", 0.5f);
        std::string on_success = cfg["on_success"] ? cfg["on_success"].as<std::string>() : "Velocity";
        std::string on_timeout = cfg["on_timeout"] ? cfg["on_timeout"].as<std::string>() : "Passive";

        if (FSMStringMap.right.count(on_success)) {
            registered_checks.emplace_back(std::make_pair(
                [this]()->bool { return standing_.load(); },
                FSMStringMap.right.at(on_success)));
        } else {
            spdlog::warn("State_{}: on_success state '{}' not found: it will not hand over",
                         state_string, on_success);
        }
        if (!FSMStringMap.right.count(on_timeout)) {
            spdlog::warn("State_{}: on_timeout state '{}' not found, using Passive",
                         state_string, on_timeout);
            on_timeout = "Passive";
        }
        registered_checks.emplace_back(std::make_pair(
            [this]()->bool { return timed_out_.load(); },
            FSMStringMap.right.at(on_timeout)));
    }

    void enter()
    {
        // Limp (damping only) until the robot lies still.
        auto & cmd = lowcmd->msg_.motor_cmd();
        for (size_t i = 0; i < kd_.size() && i < cmd.size(); ++i)
        {
            cmd[i].kp() = 0;
            cmd[i].kd() = kd_[i];
            cmd[i].dq() = 0;
            cmd[i].tau() = 0;
        }
        standing_ = false;
        timed_out_ = false;
        policy_on_ = false;
        still_time_ = 0.0f;
        t_enter_ = last_ = std::chrono::steady_clock::now();
        spdlog::info("GetUp: waiting until the robot lies still ({} s, at most {} s)",
                     settle_s_, max_settle_s_);
    }

    void run()
    {
        if (!policy_on_)
        {
            settle();
            return;
        }
        std::vector<float> a;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            a = actions_;
        }
        auto & ids = env_->robot->data.joint_ids_map;
        for (size_t i = 0; i < ids.size() && i < a.size(); ++i) {
            lowcmd->msg_.motor_cmd()[static_cast<int>(ids[i])].q() = a[i];
        }
    }

    void exit()
    {
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        spdlog::info("GetUp: {}", standing_ ? "standing, handing over" :
                                  timed_out_ ? "not up in time" : "stopped");
        policy_on_ = false;
    }

private:
    // Before the policy: motors limp, targets = where the joints are; start the policy
    // once still for settle_s_ (or after max_settle_s_).
    void settle()
    {
        auto & cmd = lowcmd->msg_.motor_cmd();
        auto & ms = lowstate->msg_.motor_state();
        float max_dq = 0.0f;
        for (size_t i = 0; i < cmd.size() && i < ms.size(); ++i)
        {
            cmd[i].q() = ms[i].q();
            max_dq = std::max(max_dq, std::fabs(ms[i].dq()));
        }
        auto & g = lowstate->msg_.imu_state().gyroscope();
        const float gyro = std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        const auto now = std::chrono::steady_clock::now();
        still_time_ = (gyro < still_gyro_ && max_dq < still_joint_vel_)
                      ? still_time_ + std::chrono::duration<float>(now - last_).count() : 0.0f;
        last_ = now;
        const float waited = std::chrono::duration<float>(now - t_enter_).count();
        if (still_time_ >= settle_s_ || waited >= max_settle_s_)
        {
            if (still_time_ < settle_s_) {
                spdlog::warn("GetUp: not still after {} s, starting anyway", max_settle_s_);
            }
            start_policy();
        }
    }

    void start_policy()
    {
        auto & data = env_->robot->data;
        for (size_t i = 0; i < data.joint_ids_map.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[static_cast<int>(data.joint_ids_map[i])];
            motor.kp() = data.joint_stiffness[i];
            motor.kd() = data.joint_damping[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        {
            std::lock_guard<std::mutex> lock(mtx_);
            env_->reset();
            env_->step();          // first targets ready before run() writes them
            actions_ = env_->action_manager->processed_actions();
        }
        stand_time_ = 0.0f;
        next_log_s_ = 0.0;
        running_ = true;
        thread_ = std::thread([this] { loop(); });
        policy_on_ = true;
        spdlog::info("GetUp: policy running (standing for {} s after at least {} s -> hand over; "
                     "not up after {} s -> give up)", hold_s_, min_time_s_, max_time_s_);
    }

    static float read(const YAML::Node & cfg, const char * key, float fallback)
    {
        return cfg[key] ? cfg[key].as<float>() : fallback;
    }

    void loop()
    {
        using clock = std::chrono::steady_clock;
        const auto dt = std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(env_->step_dt));
        const auto t0 = clock::now();
        auto next = t0 + dt;
        while (running_)
        {
            std::this_thread::sleep_until(next);
            next += dt;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                env_->step();
                actions_ = env_->action_manager->processed_actions();
            }
            status(std::chrono::duration<double>(clock::now() - t0).count());
        }
    }

    // In the policy thread, after env_->step() refreshed env_->robot->data.
    void status(double elapsed)
    {
        auto & d = env_->robot->data;
        const float gz = std::clamp(-d.projected_gravity_b[2], -1.0f, 1.0f);
        const float tilt = std::acos(gz);
        const float gyro = d.root_ang_vel_b.norm();
        float leg_dev = 0.0f;
        const int legs = std::min<int>(12, static_cast<int>(d.joint_pos.size()));
        for (int i = 0; i < legs; ++i) {
            leg_dev = std::max(leg_dev, std::fabs(d.joint_pos[i] - d.default_joint_pos[i]));
        }
        const bool up = tilt < max_tilt_ && gyro < max_gyro_ && leg_dev < max_leg_dev_;
        stand_time_ = up ? stand_time_ + env_->step_dt : 0.0f;
        if (elapsed >= min_time_s_ && stand_time_ >= hold_s_) {
            standing_ = true;
        } else if (elapsed >= max_time_s_) {
            timed_out_ = true;
        }
        if (elapsed >= next_log_s_) {
            next_log_s_ += 1.0;
            spdlog::info("GetUp: {:4.1f} s  tilt {:5.1f} deg  pelvis rotation {:4.2f} rad/s  "
                         "legs off the walking pose by up to {:5.1f} deg  {}",
                         elapsed, tilt * 57.2958f, gyro, leg_dev * 57.2958f,
                         up ? "UP" : "-");
        }
    }

    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env_;
    std::thread thread_;
    std::mutex mtx_;
    std::vector<float> actions_;
    std::atomic<bool> running_{false};
    std::atomic<bool> standing_{false};
    std::atomic<bool> timed_out_{false};
    float stand_time_ = 0.0f;
    double next_log_s_ = 0.0;

    // waiting until still (FSM thread)
    std::vector<float> kd_;
    bool policy_on_ = false;
    float still_time_ = 0.0f;
    std::chrono::steady_clock::time_point t_enter_, last_;
    float settle_s_, max_settle_s_, still_gyro_, still_joint_vel_;

    float min_time_s_, hold_s_, max_time_s_, max_tilt_, max_gyro_, max_leg_dev_;
};

REGISTER_FSM(State_GetUp)
