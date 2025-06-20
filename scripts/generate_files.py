import csv
import os
import random

def parse_arena_size(size_str):
    return tuple(map(int, size_str.strip().split('x')))

def generate_arena_3d_map(config_row, output_filename):
    width, height, layers = parse_arena_size(config_row['arena_size'])

    with open(output_filename, 'w') as f:
        f.write("type octile\n")
        f.write(f"height {height}\n")
        f.write(f"width {width}\n")
        f.write(f"layers {layers}\n")
        f.write("map\n")
        for z in range(layers):
            f.write(f"########## LAYER {z} ##########\n")
            for _ in range(height):
                f.write("." * width + "\n")

    print(f"{output_filename} 파일이 생성되었습니다.")

def generate_sample_3d_txt(config_row, output_filename):
    agent_count = int(config_row['agents'])
    width, height, layers = parse_arena_size(config_row['arena_size'])
    abs_map_path = os.path.abspath(config_row['arena_3d.map3d'])

    with open(output_filename, 'w') as f:
        f.write(f"map_file={abs_map_path}\n")
        f.write(f"agents={agent_count}\n")

        for i in range(1, agent_count + 1):
            start_key = f"start_{i}"
            goal_key = f"goal_{i}"

            start_raw = config_row.get(start_key)
            goal_raw = config_row.get(goal_key)

            # None일 경우를 대비해 빈 문자열로 대체 후 strip
            start_raw = (start_raw or "").strip()
            goal_raw = (goal_raw or "").strip()

            # 'False', '', 'None' 등은 무시하고 랜덤으로 생성
            if start_raw.lower() not in ["", "none", "false"]:
                start = start_raw.strip("()").replace(",", " ")
            else:
                start = f"{random.randint(0, width - 1)} {random.randint(0, height - 1)} {random.randint(0, layers - 1)}"

            if goal_raw.lower() not in ["", "none", "false"]:
                goal = goal_raw.strip("()").replace(",", " ")
            else:
                goal = f"{random.randint(0, width - 1)} {random.randint(0, height - 1)} {random.randint(0, layers - 1)}"

            f.write(f"{start}  {goal}\n")

    print(f"{output_filename} 파일이 생성되었습니다.")


# 메인 실행
try:
    with open('instance_config.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            generate_arena_3d_map(row, row['arena_3d.map3d'])
            generate_sample_3d_txt(row, 'sample_3d.txt')
except Exception as e:
    print("파일 생성 중 오류가 발생했습니다:", e)
