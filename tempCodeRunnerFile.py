"""
Simple script to read and display NPZ file contents.
"""

import numpy as np
from pathlib import Path


def read_npz_file(file_path, output_file=None):
    """
    Read and display information about an NPZ file.
    
    Args:
        file_path (str or Path): Path to the NPZ file
        output_file (str or Path): Optional path to save output to a text file
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        error_msg = f"❌ File not found: {file_path}"
        print(error_msg)
        return
    
    # Collect output lines
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"📦 Reading NPZ file: {file_path.name}")
    lines.append(f"{'='*80}\n")
    
    # Load the NPZ file
    data = np.load(file_path)
    
    # Display all keys
    lines.append("🔑 Keys in file:")
    for key in sorted(data.keys()):
        lines.append(f"   - {key}")
    
    # Display shape and dtype for each array
    lines.append(f"\n📊 Arrays info:")
    for key in sorted(data.keys()):
        arr = data[key]
        lines.append(f"   {key:30s} | Shape: {str(arr.shape):20s} | dtype: {arr.dtype}")
    
    # Display sample data
    lines.append(f"\n📈 Sample data (first 3 instances):")
    
    # Determine how many instances to show
    first_arr = data[list(data.keys())[0]]
    num_instances = first_arr.shape[0]
    num_to_show = min(3, num_instances)
    
    for inst_idx in range(num_to_show):
        lines.append(f"\n   === Instance {inst_idx} ===")
        for key in sorted(data.keys()):
            arr = data[key]
            if arr.size > 0:
                lines.append(f"   {key}:")
                if arr.ndim == 1:
                    lines.append(f"      {arr[inst_idx]}")  # Show single value or first N
                else:
                    lines.append(f"      shape={arr[inst_idx].shape}, sample={arr[inst_idx].flat[:5]}")  # Show shape + first 5 elements
    
    lines.append(f"\n{'='*80}\n")
    
    # Output to file or console
    output_text = "\n".join(lines)
    
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_text)
        print(f"✅ Output saved to: {output_file}")
    else:
        print(output_text)


if __name__ == "__main__":
    # Ví dụ: đọc file train
    import sys
    
    # Nếu không truyền argument, sử dụng file mặc định
    if len(sys.argv) > 1:
        file_to_read = sys.argv[1]
        output_txt = sys.argv[2] if len(sys.argv) > 2 else f"npz_info_{Path(file_to_read).stem}.txt"
    else:
        # Chọn một file từ các folder có sẵn
        file_to_read = "data/train_data_npz/pvrpwdp_epoch_00.npz"
        output_txt = "npz_info.txt"
        # file_to_read = "data/old.npz"
        # output_txt = "old.txt"
    
    read_npz_file(file_to_read, output_txt)
