"""
    read_mem_linux()

Read memory usage from /proc/meminfo on Linux.
"""
function read_mem_linux()
    if Sys.islinux()
        lines = readlines("/proc/meminfo")
        # total memory in the system
        mtot = parse(Int64, match(r"MemTotal:\s+(\d+)\skB", lines[1]).captures[1])
        # memory available
        mavail = parse(Int64, match(r"MemAvailable:\s+(\d+)\skB", lines[3]).captures[1])
        # total swap in system
        stot = parse(Int64, match(r"SwapTotal:\s+(\d+)\skB", lines[15]).captures[1])
        # free swap
        sfree = parse(Int64, match(r"SwapFree:\s+(\d+)\skB", lines[16]).captures[1])
        @printf("Of %.2f GB total memory, %.2f GB available\n", mtot / 1e6, mavail / 1e6)
        @printf("Of %.2f GB total swap, %.2f GB free\n", stot / 1e6, sfree / 1e6)
    end
end

"""
    timer_help(start)

Given start-time, return time passed in minutes and seconds
"""
function timer_help(start)
    elapsed_time = now() - start
    elapsed_seconds_total = Dates.value(elapsed_time) / 1000
    elapsed_minutes = trunc(elapsed_seconds_total / 60)
    elapsed_seconds = round(elapsed_seconds_total % 60)

    @printf(
        "Elapsed time: %02d minutes and %02d seconds\n",
        elapsed_minutes,
        elapsed_seconds
    )
end
