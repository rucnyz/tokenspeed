from tokenspeed_scheduler import Scheduler, SchedulerConfig, XPoolFirePlan


cfg = SchedulerConfig()
cfg.max_batch_size = 4
cfg.page_size = 16
cfg.num_device_pages = 49        # 32 profiled + 16 headroom + 1 reserved
cfg.enable_mamba = True
cfg.mamba_pool_total_chunks = 68 # 64 profiled + 4 headroom
cfg.mamba_cache_chunk_size = 16
cfg.kv_bytes_per_page = 4096
cfg.mamba_bytes_per_slot = 16384
cfg.enable_xpool_dynamic_capacity = True
cfg.xpool_initial_kv_pages = 32
cfg.xpool_initial_mamba_slots = 64
s = Scheduler(cfg)
print(f"KV:    mapped={s.mapped_kv_pages()}  avail={s.available_kv_pages()}")
print(f"mamba: mapped={s.mapped_mamba_slots()} avail={s.available_mamba_slots()}")
p = XPoolFirePlan()
p.direction, p.page_ids, p.op_id = "mamba_to_kv", list(range(1, 5)), 1
s.apply_xpool_fire(p)
assert s.mapped_kv_pages() == 36 and s.mapped_mamba_slots() == 63, "FAIL"
p2 = XPoolFirePlan()
p2.direction, p2.page_ids, p2.op_id = "kv_to_mamba", list(range(1, 5)), 2
s.apply_xpool_fire(p2)
assert s.mapped_kv_pages() == 32 and s.mapped_mamba_slots() == 64, "FAIL"
print("apply_xpool_fire round-trip: PASS")