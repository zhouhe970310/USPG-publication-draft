#!/usr/bin/env python
"""实时监控训练进度"""
import time
import os

output_file = "/private/tmp/claude-501/-Users-yuhuazhao-Downloads-CV/b7d83594-90d3-4a9f-b1e6-9a18b981e6ae/tasks/bsaha0jl9.output"

print("=" * 80)
print("USPG-Net 训练进度监控")
print("=" * 80)
print(f"监控文件: {output_file}")
print("按 Ctrl+C 停止监控")
print("=" * 80)
print()

last_size = 0
try:
    while True:
        if os.path.exists(output_file):
            current_size = os.path.getsize(output_file)
            if current_size > last_size:
                with open(output_file, 'r') as f:
                    f.seek(last_size)
                    new_content = f.read()
                    # 只显示包含重要信息的行
                    for line in new_content.split('\n'):
                        if any(keyword in line for keyword in ['Epoch', 'Train Loss', 'Val', 'Dice', 'Best', 'completed']):
                            print(line)
                last_size = current_size
        time.sleep(5)
except KeyboardInterrupt:
    print("\n监控已停止")
