"""
将 Jinan 路网的清空相位格式转换为 Test 路网格式

转换规则：
- 's' 状态 (5秒) → 'y' 状态 (3秒) + 'r' 状态 (2秒)
- 为清空相位添加 name="YELLOW_ALL_RED" 属性
- 保持绿灯相位不变
"""

import xml.etree.ElementTree as ET
import os
import shutil
from datetime import datetime


def convert_jinan_to_test_format(input_path, output_path, backup=True):
    """
    将 Jinan 格式的 TLL 文件转换为 Test 格式
    
    Args:
        input_path: 输入的 jinan.tll.xml 路径
        output_path: 输出的新文件路径
        backup: 是否备份原文件
    """
    
    # Implementation note.
    if backup and os.path.exists(input_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.replace('.xml', f'_backup_{timestamp}.xml')
        shutil.copy2(input_path, backup_path)
        print(f"✓ 已备份原文件到: {backup_path}")
    
    # Implementation note.
    tree = ET.parse(input_path)
    root = tree.getroot()
    
    total_conversions = 0
    
    # Implementation note.
    for tl_logic in root.findall('tlLogic'):
        tls_id = tl_logic.get('id')
        print(f"\n处理路口: {tls_id}")
        
        new_phases = []
        phase_count = 0
        
        for phase in tl_logic.findall('phase'):
            state = phase.get('state')
            duration = phase.get('duration')
            name = phase.get('name', '')
            
            # Implementation note.
            if 's' in state:
                # Implementation note.
                yellow_state = state.replace('s', 'y')
                yellow_phase = ET.Element('phase')
                yellow_phase.set('duration', '3')
                yellow_phase.set('state', yellow_state)
                yellow_phase.set('name', 'YELLOW_ALL_RED')
                new_phases.append(yellow_phase)
                
                # Implementation note.
                red_state = state.replace('s', 'r')
                red_phase = ET.Element('phase')
                red_phase.set('duration', '2')
                red_phase.set('state', red_state)
                new_phases.append(red_phase)
                
                total_conversions += 1
                phase_count += 2
                print(f"  ✓ 转换清空相位: 's'(5s) → 'y'(3s) + 'r'(2s)")
            else:
                # Implementation note.
                new_phase = ET.Element('phase')
                new_phase.set('duration', duration)
                new_phase.set('state', state)
                if name:
                    new_phase.set('name', name)
                new_phases.append(new_phase)
                phase_count += 1
        
        # Implementation note.
        for phase in list(tl_logic):
            tl_logic.remove(phase)
        
        for phase in new_phases:
            tl_logic.append(phase)
        
        print(f"  总相位数: {phase_count}")
    
    # Implementation note.
    # Implementation note.
    indent_xml(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    
    print(f"\n{'='*70}")
    print(f"转换完成！")
    print(f"{'='*70}")
    print(f"总共转换了 {total_conversions} 个清空相位")
    print(f"输出文件: {output_path}")
    
    return total_conversions


def indent_xml(elem, level=0):
    """格式化 XML 输出，使其更易读"""
    indent = "\n" + "\t" * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "\t"
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def verify_conversion(original_path, converted_path):
    """验证转换结果"""
    print(f"\n{'='*70}")
    print("验证转换结果")
    print(f"{'='*70}")
    
    orig_tree = ET.parse(original_path)
    conv_tree = ET.parse(converted_path)
    
    orig_root = orig_tree.getroot()
    conv_root = conv_tree.getroot()
    
    orig_tls = orig_root.findall('tlLogic')
    conv_tls = conv_root.findall('tlLogic')
    
    print(f"原文件路口数: {len(orig_tls)}")
    print(f"转换后路口数: {len(conv_tls)}")
    
    if len(orig_tls) != len(conv_tls):
        print("⚠️ 警告: 路口数量不匹配！")
        return False
    
    for orig_tl, conv_tl in zip(orig_tls, conv_tls):
        tls_id = orig_tl.get('id')
        orig_phases = orig_tl.findall('phase')
        conv_phases = conv_tl.findall('phase')
        
        # Implementation note.
        # Implementation note.
        s_count = sum(1 for p in orig_phases if 's' in p.get('state'))
        expected_count = len(orig_phases) + s_count
        
        if len(conv_phases) != expected_count:
            print(f"⚠️ {tls_id}: 相位数不匹配")
            print(f"   原始: {len(orig_phases)}, 转换后: {len(conv_phases)}, 预期: {expected_count}")
            return False
    
    print("✓ 验证通过！所有路口的相位数量正确")
    
    # Implementation note.
    yellow_count = 0
    for tl in conv_tls:
        for phase in tl.findall('phase'):
            if phase.get('name') == 'YELLOW_ALL_RED':
                yellow_count += 1
    
    print(f"✓ 找到 {yellow_count} 个 YELLOW_ALL_RED 相位")
    
    return True


if __name__ == "__main__":
    # Implementation note.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    jinan_dir = os.path.join(project_root, 'data', 'Jinan', '3_4')
    
    input_file = os.path.join(jinan_dir, 'jinan.tll.xml')
    output_file = os.path.join(jinan_dir, 'jinan.tll.xml')  # Implementation note.
    
    print("="*70)
    print("Jinan 路网清空相位格式转换工具")
    print("="*70)
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    
    if not os.path.exists(input_file):
        print(f"\n❌ 错误: 找不到输入文件 {input_file}")
        exit(1)
    
    # Implementation note.
    conversions = convert_jinan_to_test_format(input_file, output_file, backup=True)
    
    # Implementation note.
    backup_files = [f for f in os.listdir(jinan_dir) if 'backup' in f and f.endswith('.xml')]
    if backup_files:
        latest_backup = os.path.join(jinan_dir, sorted(backup_files)[-1])
        verify_conversion(latest_backup, output_file)
    
    print(f"\n{'='*70}")
    print("转换完成！现在 Jinan 路网使用与 Test 路网相同的清空机制：")
    print("  - 黄灯阶段: 3秒 (状态 'y')")
    print("  - 全红阶段: 2秒 (状态 'r')")
    print("  - 清空相位命名: YELLOW_ALL_RED")
    print(f"{'='*70}")
