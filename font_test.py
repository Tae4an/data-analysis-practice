#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한글 폰트 테스트 스크립트
matplotlib에서 한글이 제대로 표시되는지 확인합니다.
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import numpy as np

print("="*50)
print("한글 폰트 테스트")
print("="*50)

# 사용 가능한 한글 폰트 찾기
korean_fonts = ['Apple SD Gothic Neo', 'Nanum Gothic', 'AppleGothic', 'Malgun Gothic', 'Dotum']

available_font = None
for font_name in korean_fonts:
    if font_name in [f.name for f in fm.fontManager.ttflist]:
        available_font = font_name
        break

if available_font:
    plt.rcParams['font.family'] = available_font
    print(f"✅ 한글 폰트 설정됨: {available_font}")
else:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    print("❌ 한글 폰트를 찾을 수 없습니다.")

plt.rcParams['axes.unicode_minus'] = False

# 간단한 한글 그래프 테스트
print("\n간단한 한글 그래프를 생성합니다...")

# 테스트 데이터
categories = ['사과', '바나나', '오렌지', '포도', '딸기']
values = [10, 25, 30, 25, 20]

# 그래프 생성
plt.figure(figsize=(10, 6))
bars = plt.bar(categories, values, color=['red', 'yellow', 'orange', 'purple', 'pink'])

# 한글 제목과 레이블
plt.title('과일 판매량 차트', fontsize=16, fontweight='bold')
plt.xlabel('과일 종류', fontsize=12)
plt.ylabel('판매량 (개)', fontsize=12)

# 각 막대 위에 값 표시
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{value}개', ha='center', va='bottom', fontsize=10)

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# 파일로 저장
plt.savefig('korean_font_test.png', dpi=300, bbox_inches='tight')
print("✅ 그래프가 'korean_font_test.png'로 저장되었습니다.")

# 화면에 표시
plt.show()

print("\n" + "="*50)
print("폰트 테스트 완료!")
print("그래프에서 한글이 제대로 보이는지 확인해주세요.")
print("만약 여전히 네모로 보인다면 다른 폰트로 변경이 필요합니다.")
print("="*50) 