import csv

with open("result.txt", "r") as infile:
    lines = infile.readlines()

with open("result.csv", "w", newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["agent", "time", "x", "y", "z"])  # 헤더

    is_solution = False

    for line in lines:
        line = line.strip()

        # solution= 다음부터 처리 시작
        if line == "solution=":
            is_solution = True
            continue

        if is_solution and line:
            if ':' not in line:
                continue

            time_str, positions_str = line.split(":")
            time = int(time_str.strip())

            # 에이전트 별 위치 추출
            positions = [p.strip() for p in positions_str.split('),') if p]
            for agent_id, pos in enumerate(positions):
                pos = pos.strip()
                if not pos.endswith(")"):
                    pos += ")"
                if pos.startswith("(") and pos.endswith(")"):
                    try:
                        # 좌표를 float으로 파싱
                        x, y, z = map(float, pos.strip("()").split(","))
                        writer.writerow([agent_id, time, x, y, z])
                    except Exception as e:
                        print("Error parsing position:", pos, e)
