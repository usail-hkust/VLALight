"""
修复 BERT 模型目录结构
将 HuggingFace 缓存格式转换为 transformers 期望的直接结构
"""

import os
import shutil
from pathlib import Path

def fix_bert_structure():
    """将 BERT 从 HuggingFace 缓存结构转换为直接结构"""
    project_root = Path(__file__).parent.parent
    weights_dir = project_root / "weights"
    
    # Implementation note.
    cache_bert_dir = weights_dir / "bert-base-uncased"
    snapshots_dir = cache_bert_dir / "snapshots"
    
    print(f"🔧 修复 BERT 目录结构")
    print(f"📂 当前路径: {cache_bert_dir}")
    
    # Implementation note.
    snapshot_hash = None
    if snapshots_dir.exists():
        for item in snapshots_dir.iterdir():
            if item.is_dir() and len(item.name) == 40:  # SHA hash
                snapshot_hash = item.name
                break
    
    if not snapshot_hash:
        print(f"❌ 找不到 BERT snapshot 目录")
        return False
    
    snapshot_path = snapshots_dir / snapshot_hash
    print(f"📦 找到 snapshot: {snapshot_hash}")
    
    # Implementation note.
    new_bert_dir = weights_dir / "bert-base-uncased-direct"
    
    if new_bert_dir.exists():
        print(f"🗑️ 删除现有目录: {new_bert_dir}")
        shutil.rmtree(new_bert_dir)
    
    new_bert_dir.mkdir()
    
    try:
        # Implementation note.
        print(f"📋 复制 BERT 文件到新结构...")
        
        for file_path in snapshot_path.iterdir():
            if file_path.is_file():
                dest_path = new_bert_dir / file_path.name
                shutil.copy2(file_path, dest_path)
                file_size = file_path.stat().st_size / 1024 / 1024
                print(f"  ✅ {file_path.name} ({file_size:.1f} MB)")
        
        # Implementation note.
        print(f"🗑️ 删除旧缓存结构...")
        shutil.rmtree(cache_bert_dir)
        
        # Implementation note.
        new_bert_dir.rename(cache_bert_dir)
        
        print(f"✅ BERT 结构修复完成!")
        print(f"📍 新路径: {cache_bert_dir}")
        
        # Implementation note.
        print(f"\n🔍 验证新结构:")
        for file_path in cache_bert_dir.iterdir():
            if file_path.is_file():
                file_size = file_path.stat().st_size / 1024 / 1024
                print(f"  📄 {file_path.name} ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

if __name__ == "__main__":
    print("🛠️ BERT 结构修复工具")
    print("=" * 40)
    
    success = fix_bert_structure()
    
    if success:
        print("\n🎉 BERT 结构修复完成!")
        print("💡 现在可以正常加载 BERT tokenizer")
    else:
        print("\n❌ BERT 结构修复失败")
