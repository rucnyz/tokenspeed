#pragma once

#include <algorithm>
#include <cstdint>

#include "scheduler/types.h"

namespace tokenspeed {

class AdmissionController {
public:
    explicit AdmissionController(const AdmissionConfig& config, std::int32_t total_pages)
        : config_{config}, usable_pages_{total_pages - 1} {}

    bool ShouldAdmit(std::int32_t available_pages, std::int32_t estimated_cost) const {
        std::int32_t headroom = static_cast<std::int32_t>(usable_pages_ * (1.0 - config_.high_watermark));
        return available_pages >= estimated_cost + headroom;
    }
    const AdmissionConfig& Config() const { return config_; }

private:
    AdmissionConfig config_;
    std::int32_t usable_pages_;
};

}  // namespace tokenspeed
