// State_Fallen: g1_ctrl state for after a fall. The motors go limp (damping only, like
// Passive) so the robot lands softly and comes to rest; then the get-up state starts,
// either by itself once the robot has been still for settle_s (auto_getup: true) or when
// the operator presses the get-up buttons (a transition in config.yaml).
//
// "Still": pelvis angular speed below still_gyro and every motor slower than
// still_joint_vel.
//
// config.yaml (FSM section), e.g.:
//   Fallen:
//     transitions:
//       Passive: LT + B.on_pressed
//       GetUp: RT + X.on_pressed
//     auto_getup: false     # true: get up by itself once still
//     settle_s: 1.0         # s
//     still_gyro: 0.5       # rad/s
//     still_joint_vel: 1.0  # rad/s
//     getup_state: GetUp
//     # kd: [...]           # damping per motor; default: the Passive state's
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

#include "FSM/FSMState.h"

class State_Fallen : public FSMState
{
public:
    State_Fallen(int state, std::string state_string)
    : FSMState(state, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        kd_ = (cfg["kd"] ? cfg["kd"] : param::config["FSM"]["Passive"]["kd"]).as<std::vector<float>>();
        auto_getup_ = cfg["auto_getup"] ? cfg["auto_getup"].as<bool>() : false;
        settle_s_ = cfg["settle_s"] ? cfg["settle_s"].as<float>() : 1.0f;
        still_gyro_ = cfg["still_gyro"] ? cfg["still_gyro"].as<float>() : 0.5f;
        still_joint_vel_ = cfg["still_joint_vel"] ? cfg["still_joint_vel"].as<float>() : 1.0f;
        std::string next = cfg["getup_state"] ? cfg["getup_state"].as<std::string>() : "GetUp";

        if (auto_getup_) {
            if (FSMStringMap.right.count(next)) {
                registered_checks.emplace_back(std::make_pair(
                    [this]()->bool { return still_time_ >= settle_s_; },
                    FSMStringMap.right.at(next)));
            } else {
                spdlog::warn("State_{}: getup_state '{}' not found: no automatic get-up",
                             state_string, next);
                auto_getup_ = false;
            }
        }
    }

    void enter()
    {
        auto & cmd = lowcmd->msg_.motor_cmd();
        for (size_t i = 0; i < kd_.size() && i < cmd.size(); ++i)
        {
            cmd[i].kp() = 0;
            cmd[i].kd() = kd_[i];
            cmd[i].dq() = 0;
            cmd[i].tau() = 0;
        }
        still_time_ = 0.0f;
        last_ = std::chrono::steady_clock::now();
        spdlog::info("Fallen: motors limp (damping only). {}",
                     auto_getup_ ? "Getting up once still." : "Get up with the GetUp buttons.");
    }

    void run()
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
        const float dt = std::chrono::duration<float>(now - last_).count();
        last_ = now;
        still_time_ = (gyro < still_gyro_ && max_dq < still_joint_vel_) ? still_time_ + dt : 0.0f;
    }

private:
    std::vector<float> kd_;
    bool auto_getup_ = false;
    float settle_s_ = 1.0f;
    float still_gyro_ = 0.5f;
    float still_joint_vel_ = 1.0f;
    float still_time_ = 0.0f;
    std::chrono::steady_clock::time_point last_;
};

REGISTER_FSM(State_Fallen)
