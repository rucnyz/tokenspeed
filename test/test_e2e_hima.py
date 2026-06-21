from tokenspeed_scheduler import SchedulerConfig

cfg = SchedulerConfig()
# Phase 1
cfg.eviction_policy = "lpb"
cfg.lpb_window_s = 60.0
cfg.lpb_hit_deque_maxlen = 4096
cfg.c_kv_alpha = 1.02e-7
cfg.c_kv_beta = 0.0246
cfg.c_kv_gamma = 5.97
cfg.c_m = 0.0
# Phase 2
cfg.enable_budgeter = True
cfg.enable_admitter = True
cfg.enable_xpool_dynamic_capacity = True
cfg.budgeter_pages_per_fire = 64
cfg.kv_bytes_per_page = 131072
cfg.mamba_bytes_per_slot = 16384
# XPoolFirePlan
from tokenspeed_scheduler import XPoolFirePlan

plan = XPoolFirePlan()
plan.direction = "mamba_to_kv"
plan.page_ids = [1, 2, 3]
print("SchedulerConfig + XPoolFirePlan is OK")
