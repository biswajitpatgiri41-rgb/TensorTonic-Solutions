import heapq

def schedule_pipeline(tasks, resource_budget):
    task_map = {t["name"]: t for t in tasks}
    completed = set()
    started = set()
    running = []
    schedule = []
    current_time = 0
    used_resources = 0

    while len(completed) < len(tasks):
        while running and running[0][0] <= current_time:
            end_time, name = heapq.heappop(running)
            used_resources -= task_map[name]["resources"]
            completed.add(name)

        ready = []
        for t in tasks:
            name = t["name"]
            if name not in started and all(dep in completed for dep in t["depends_on"]):
                ready.append(t)
        ready.sort(key=lambda x: x["name"])

        scheduled = False
        for t in ready:
            if used_resources + t["resources"] <= resource_budget:
                started.add(t["name"])
                schedule.append((t["name"], current_time))
                used_resources += t["resources"]
                heapq.heappush(running, (current_time + t["duration"], t["name"]))
                scheduled = True

        if len(completed) == len(tasks):
            break

        if running:
            current_time = running[0][0]
        elif not scheduled:
            break

    return sorted(schedule, key=lambda x: (x[1], x[0]))