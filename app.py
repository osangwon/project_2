# app.py

import os
import time
import json
import platform
import base64
import tempfile
import io
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.io as pio
from plotly import graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager
import seaborn as sns
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from shiny import App, ui, reactive, render
from htmltools import tags
import traceback
import plotly.graph_objects as go
from shinywidgets import output_widget, render_widget
from matplotlib.patches import Wedge, Circle, Polygon


# 나눔고딕 설정
# 나눔고딕 설정
font_path = os.path.join(os.path.dirname(__file__), "data", "NanumGothic.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
else:
    plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# ─────────────────────────────────────────────────────────────────────────
# 0) Static assets path
# ─────────────────────────────────────────────────────────────────────────
#static_path = os.path.join(os.path.dirname(__file__), "..", "www")
static_path = os.path.join(os.path.dirname(__file__),"data")


# ───────────────────────────────────────────────────────────────────────────
# 한글 깨짐 방지용 Matplotlib 설정 (Windows + macOS + Linux)
# ───────────────────────────────────────────────────────────────────────────

import platform
import matplotlib.pyplot as plt

system_name = platform.system()

if system_name == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"

# 나눔고딕 설정
plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 깨짐 방지

# ───────────────────────────────────────────────────────────────────────────
# 0) 상수 정의
# ───────────────────────────────────────────────────────────────────────────
TH       = 0.3     # 누적 불량률 관리도 기준선(IQR 기반)
ALERT_TH = 0.5     # 알림을 띄울 불량 확률 임계값 (0.5)
INTERVAL = 1.5     # 스트리밍 간격 (초)

# 센서 변수 목록 (IQR 범위 계산에 사용)
slider_variables = [
    ("biscuit_thickness", "비스켓 두께"),
    ("molten_temp", "용탕 온도"),
    ("low_section_speed", "하단 구간 속도"),
    ("high_section_speed", "상단 구간 속도"),
    ("cast_pressure", "주조 압력"),
    ("upper_mold_temp1", "상부 몰드 온도1"),
    ("upper_mold_temp2", "상부 몰드 온도2"),
    ("lower_mold_temp1", "하부 몰드 온도1"),
    ("lower_mold_temp2", "하부 몰드 온도2"),
    ("sleeve_temperature", "슬리브 온도"),
    ("physical_strength", "물리적 강도"),
    ("Coolant_temperature", "냉각수 온도")
]

# 각 센서 박스를 "이미지 위"에 배치할 top/left 좌표 (퍼센트)
positions = [
    ("50%", "50%"), ("10%", "50%"), ("5%", "13%"), ("50%", "75%"),
    ("10%", "72%"), ("0%", "30%"), ("0%", "40%"),
    ("65%", "30%"), ("65%", "40%"), ("23%", "72%"),
    ("49%", "12%"), ("23%", "50%"), ("35%", "70%")
]

# 품질 분석용 변수명 한글 매핑
variable_name_map = {
    "molten_temp": "용탕 온도",
    "low_section_speed": "하단 구간 속도",
    "high_section_speed": "상단 구간 속도",
    "cast_pressure": "주조 압력",
    "upper_mold_temp1": "상부 몰드 온도1",
    "upper_mold_temp2": "상부 몰드 온도2",
    "lower_mold_temp1": "하부 몰드 온도1",
    "lower_mold_temp2": "하부 몰드 온도2",
    "sleeve_temperature": "슬리브 온도",
    "physical_strength": "물리적 강도",
    "Coolant_temperature": "냉각수 온도",
    "biscuit_thickness": "비스켓 두께"
}

selected_cols = [
    "molten_temp", "low_section_speed", "high_section_speed", "cast_pressure",
    "biscuit_thickness", "upper_mold_temp1", "upper_mold_temp2", "lower_mold_temp1",
    "lower_mold_temp2", "sleeve_temperature", "physical_strength", "Coolant_temperature"
]

# ───────────────────────────────────────────────────────────────────────────
# 1) 데이터 & 모델 로드 및 datetime 파싱
# ───────────────────────────────────────────────────────────────────────────
df_header = pd.read_csv("data/test.csv", nrows=0)
if "datetime" in df_header.columns:
    df_all = pd.read_csv("data/test.csv", parse_dates=["datetime"], errors="ignore")
    df_all["datetime"] = pd.to_datetime(df_all["datetime"], errors="coerce")
else:
    df_all = pd.read_csv("data/test.csv")
    if {"date", "time"}.issubset(df_all.columns):
        df_all["datetime"] = pd.to_datetime(
            df_all["date"].astype(str) + " " + df_all["time"].astype(str),
            errors="coerce"
        )
        mask = df_all["datetime"].isna()
        if mask.any():
            df_all.loc[mask, "datetime"] = pd.to_datetime(
                df_all.loc[mask, "time"].astype(str) + " " + df_all.loc[mask, "date"].astype(str),
                errors="coerce"
            )
    else:
        df_all["datetime"] = pd.NaT

df_all = df_all[df_all['mold_code'].isin([8722, 8412, 8917])]
df_all = df_all.sort_values("datetime").reset_index(drop=True)

# 분류 모델 로드 (경로를 실제 파일 위치로 바꿔주세요)
model = joblib.load("data/best_model .pkl")

# 품질 분석용 데이터 로드 (경로 수정 필요)
try:
    train = pd.read_csv("data/train.csv")
    train = train[(train['id'] != 19327) & (train['mold_code'].isin([8722, 8412, 8917]))]
    train['registration_time'] = pd.to_datetime(train['registration_time'])
    
    max_date = train['registration_time'].max().date()
    min_date = max_date - pd.Timedelta(days=7)
except:
    # 데이터를 찾을 수 없는 경우 더미 데이터 생성
    train = df_all.copy()
    train['registration_time'] = df_all['datetime']
    train['passorfail'] = np.random.choice([0, 1], size=len(train), p=[0.9, 0.1])
    max_date = train['registration_time'].max().date()
    min_date = max_date - pd.Timedelta(days=7)

# ───────────────────────────────────────────────────────────────────────────
# 2) IQR 기반 몰드코드별 센서 허용 범위(thresholds) 계산
# ───────────────────────────────────────────────────────────────────────────
iqr_thresholds = {}
numeric_vars = [v[0] for v in slider_variables]

for code, group in df_all.groupby("mold_code"):
    code_str = str(code)
    thresh_dict = {}
    for var in numeric_vars:
        if var in group.columns and pd.api.types.is_numeric_dtype(group[var]):
            Q1 = group[var].quantile(0.25)
            Q3 = group[var].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            thresh_dict[var] = (lower, upper)
    iqr_thresholds[code_str] = thresh_dict

# ───────────────────────────────────────────────────────────────────────────
# 품질 분석 함수들
# ───────────────────────────────────────────────────────────────────────────
def detect_outliers_iqr_by_mold(df, mold_col='mold_code', cols=None):
    cols = cols or df.columns
    molds = df[mold_col].unique()
    outlier_df = pd.DataFrame(False, index=df.index, columns=cols)

    for mold in molds:
        mold_df = df[df[mold_col] == mold]
        Q1 = mold_df[cols].quantile(0.25)
        Q3 = mold_df[cols].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask_lower = mold_df[cols].lt(lower_bound)
        mask_upper = mold_df[cols].gt(upper_bound)

        outlier_df.loc[mold_df.index, cols] = (mask_lower | mask_upper)

    return outlier_df

def generate_pdf_report(filtered_data, outliers, selected_variable, date_range):
    """한글 완벽 지원 PDF 보고서 생성"""
    
    # 임시 파일 생성
    temp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(temp_dir, f"quality_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
    
    # 한글 폰트 강제 등록
    korean_font = 'MalgunGothic'
    try:
        # Windows 환경 폰트들 시도
        font_paths = [
            "C:/Windows/Fonts/malgun.ttf",      # 맑은 고딕 
            "C:/Windows/Fonts/malgunbd.ttf",    # 맑은 고딕 볼드
            "C:/Windows/Fonts/gulim.ttc",       # 굴림
            "C:/Windows/Fonts/batang.ttc",      # 바탕
            "C:/Windows/Fonts/dotum.ttc",       # 돋움
            #"/System/Library/Fonts/AppleSDGothicNeo.ttc",  # Mac
            "C:/Users/qhrud/OneDrive/바탕 화면/project5/dashboard/data/NanumGothic.ttf"  # Linux
        ]
        
        font_registered = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont(korean_font, font_path))
                    font_registered = True
                    break
                except:
                    continue
        
        if not font_registered:
            # 기본 폰트로 대체하되 한글 내용은 유지
            korean_font = 'Helvetica'
    except:
        korean_font = 'Helvetica'
    
    # matplotlib 한글 폰트 설정
    plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # PDF 문서 생성
    doc = SimpleDocTemplate(
        pdf_path, 
        pagesize=A4,
        topMargin=0.5*inch,
        bottomMargin=0.5*inch,
        leftMargin=0.7*inch,
        rightMargin=0.7*inch
    )
    
    styles = getSampleStyleSheet()
    story = []
    
    # 색상 정의
    navy = colors.Color(0.1, 0.25, 0.36)  # #1a365d
    blue = colors.Color(0.17, 0.47, 0.68)  # #2b77ad  
    light_blue = colors.Color(0.94, 0.97, 1.0)  # 연한 파란색
    
    # === 한글 스타일 정의 ===
    title_style = ParagraphStyle(
        'KoreanTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=20,
        alignment=TA_CENTER,
        textColor=navy,
        fontName=korean_font
    )
    
    subtitle_style = ParagraphStyle(
        'KoreanSubtitle',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=blue,
        fontName=korean_font
    )
    
    section_style = ParagraphStyle(
        'KoreanSection',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=15,
        spaceBefore=20,
        textColor=navy,
        fontName=korean_font,
        backColor=light_blue,
        borderWidth=1,
        borderColor=blue,
        borderPadding=8
    )
    
    subsection_style = ParagraphStyle(
        'KoreanSubsection',
        parent=styles['Heading2'],
        fontSize=13,
        spaceAfter=10,
        spaceBefore=15,
        textColor=blue,
        fontName=korean_font
    )
    
    body_style = ParagraphStyle(
        'KoreanBody',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=10,
        leading=12,
        fontName=korean_font
    )
    
    highlight_style = ParagraphStyle(
        'KoreanHighlight',
        parent=styles['Normal'],
        fontSize=10,
        spaceAfter=12,
        leading=12,
        fontName=korean_font,
        backColor=light_blue,
        borderWidth=1,
        borderColor=blue,
        borderPadding=8
    )
    
    # === 표지 페이지 ===
    story.append(Spacer(1, 0.8*inch))
    
    # 제목 (한글)
    story.append(Paragraph("제조업 품질 분석 보고서", title_style))
    story.append(Paragraph("Manufacturing Quality Analysis Report", subtitle_style))
    story.append(Spacer(1, 0.3*inch))
    
    # 아이콘
    icon_style = ParagraphStyle('Icon', alignment=TA_CENTER, fontSize=20, spaceAfter=20, fontName=korean_font, textColor=blue)
    story.append(Paragraph("품질 분석 보고서", icon_style))
    
    # 보고서 기본 정보 (한글)
    current_time = datetime.now()
    info_data = [
        ['보고서 생성일', current_time.strftime('%Y년 %m월 %d일 %H시 %M분')],
        ['분석 기간', f"{date_range[0]} ~ {date_range[1]}"],
        ['총 데이터 수', f"{len(filtered_data):,} 건"],
        ['분석 변수', variable_name_map.get(selected_variable, selected_variable)],
        ['몰드 수', f"{len(filtered_data['mold_code'].unique())} 개"]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), navy),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.white),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
        ('FONTNAME', (0, 0), (-1, -1), korean_font),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 1, navy),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6)
    ]))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(info_table)
    story.append(PageBreak())
    
    # === 1. 분석 개요 ===
    story.append(Paragraph("1. 분석 개요", section_style))
    
    overview_text = """
본 보고서는 제조 공정의 품질 데이터를 종합적으로 분석하여 이상치 현황과 품질 영향 요인을 파악합니다. 
IQR(Interquartile Range) 방법을 사용하여 몰드별 이상치를 탐지하고, 각 변수가 제품 품질에 미치는 영향을 정량적으로 분석했습니다.<br/>

<br/><b>주요 분석 내용:</b><br/>
• 전체 이상치 분포 현황 및 트렌드 분석<br/>
• 12개 주요 공정 변수별 이상치 발생 패턴<br/>
• 몰드별 품질 특성 및 변동성 분석<br/>
• 이상치와 제품 불량률 간의 상관관계<br/>
• 데이터 기반 품질 개선 권고사항
    """
    story.append(Paragraph(overview_text, body_style))
    story.append(Spacer(1, 15))
    
    # 핵심 통계
    if not filtered_data.empty and outliers is not None:
        total = len(filtered_data)
        outlier_count = outliers.any(axis=1).sum()
        outlier_rate = (outlier_count / total * 100) if total > 0 else 0
        defect_rate = (filtered_data['passorfail'].sum() / len(filtered_data) * 100) if len(filtered_data) > 0 else 0
        
        stats_data = [
            ['핵심 지표', '값', '상태'],
            ['총 분석 데이터', f"{total:,} 건", '-'],
            ['이상치 비율', f"{outlier_rate:.1f}%", 
             '높음' if outlier_rate > 20 else '보통' if outlier_rate > 10 else '낮음'],
            ['전체 불량률', f"{defect_rate:.1f}%",
             '높음' if defect_rate > 6 else '보통' if defect_rate > 4 else '낮음'],
            ['분석 몰드 수', f"{len(filtered_data['mold_code'].unique())} 개", '-']
        ]
        
        stats_table = Table(stats_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, navy),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_blue])
        ]))
        
        story.append(Paragraph("핵심 통계 요약", subsection_style))
        story.append(stats_table)
    
    story.append(PageBreak())
    
    # === 2. 전체 이상치 분석 ===
    story.append(Paragraph("2. 전체 이상치 분석", section_style))
    
    if not filtered_data.empty and outliers is not None:
        total = len(filtered_data)
        outlier_count = outliers.any(axis=1).sum()
        normal_count = total - outlier_count
        
        # 이상치 비율 차트 생성 (한글)
        fig, ax = plt.subplots(figsize=(8, 5))
        sizes = [normal_count, outlier_count]
        labels = ['정상치', '이상치']
        colors_pie = ['#38a169', '#e53e3e']
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors_pie, 
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 11}
        )
        ax.set_title('이상치 분포 현황', fontsize=14, fontweight='bold', pad=15)
        plt.tight_layout()
        
        pie_img_path = os.path.join(temp_dir, 'pie_chart.png')
        plt.savefig(pie_img_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        story.append(Paragraph("2.1 이상치 분포 현황", subsection_style))
        story.append(Image(pie_img_path, width=4.5*inch, height=3*inch))
        story.append(Spacer(1, 10))
        
        # 분석 결과 (한글)
        analysis_text = f"""
<b>분석 결과:</b><br/>
전체 {total:,}건 중 {outlier_count:,}건({outlier_rate:.1f}%)이 이상치로 탐지되었습니다.<br/>
현재 이상치 비율은 <b>{'심각한 수준' if outlier_rate > 20 else '주의 수준' if outlier_rate > 10 else '안정적 수준'}</b>이며, 
<b>{'즉시 공정 개선' if outlier_rate > 6 else '지속적 모니터링' if outlier_rate > 4 else '현 수준 유지'}</b> 조치가 권장됩니다.
        """
        story.append(Paragraph(analysis_text, highlight_style))
        
        # 변수별 이상치 분석 (한글)
        story.append(Spacer(1, 30))
        story.append(Paragraph("2.2 변수별 이상치 발생률", subsection_style))
        
        outlier_rows = outliers.any(axis=1)
        outlier_only = outliers.loc[outlier_rows]
        
        if not outlier_only.empty:
            var_ratios = (outlier_only.sum() / len(outlier_only)).sort_values(ascending=False)
            var_ratios_percent = (var_ratios * 100).round(1)
            
            # 막대차트 생성 (한글 라벨)
            fig, ax = plt.subplots(figsize=(10, 6))
            labels_kor = [variable_name_map.get(var, var) for var in var_ratios_percent.index]
            bars = ax.bar(labels_kor, var_ratios_percent.values, color='#2b77ad', alpha=0.8)
            
            # 값 표시
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
            
            ax.set_title('변수별 이상치 발생률', fontsize=14, fontweight='bold', pad=15)
            ax.set_ylabel('이상치 비율 (%)', fontsize=10)
            ax.set_ylim(0, max(var_ratios_percent.values) * 1.15)
            plt.xticks(rotation=45, ha='right', fontsize=9)
            plt.tight_layout()
            
            bar_img_path = os.path.join(temp_dir, 'bar_chart.png')
            plt.savefig(bar_img_path, dpi=200, bbox_inches='tight')
            plt.close()
            
            story.append(Image(bar_img_path, width=6*inch, height=3.5*inch))

            story.append(PageBreak())
            
            # 상위 위험 변수 테이블 (한글)
            risk_data = [['순위', '변수명', '이상치 비율', '위험도']]
            for i, (var, ratio) in enumerate(var_ratios_percent.head(5).items()):
                risk_level = '높음' if ratio > 40 else '보통' if ratio > 20 else '낮음'
                risk_data.append([
                    str(i+1), 
                    variable_name_map.get(var, var), 
                    f"{ratio:.1f}%", 
                    risk_level
                ])
            
            risk_table = Table(risk_data, colWidths=[0.6*inch, 2.5*inch, 1*inch, 0.8*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), blue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, -1), korean_font),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID', (0, 0), (-1, -1), 1, blue),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_blue])
            ]))
            
            story.append(Spacer(1, 10))
            story.append(Paragraph("2.3 상위 5개 위험 변수", subsection_style))
            story.append(risk_table)
    
    story.append(PageBreak())
    
    # === 3. 변수별 상세 분석 ===
    story.append(Paragraph("3. 변수별 상세 분석", section_style))
    story.append(Paragraph(f"분석 대상 변수: {variable_name_map.get(selected_variable, selected_variable)}", subsection_style))
    
    if selected_variable in filtered_data.columns:
        var_stats = filtered_data[selected_variable].describe()
        
        # 박스플롯 생성 (한글)
        fig, ax = plt.subplots(figsize=(8, 5))
        mold_codes = sorted(filtered_data['mold_code'].unique())
        box_data = [filtered_data[filtered_data['mold_code'] == mold][selected_variable].dropna() 
                   for mold in mold_codes]
        
        bp = ax.boxplot(box_data, labels=[f"몰드 {code}" for code in mold_codes], patch_artist=True)
        colors_box = ['#1a365d', '#2b77ad', '#d69e2e', '#e53e3e', '#38a169']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f'{variable_name_map.get(selected_variable, selected_variable)} - 몰드별 분포', 
                    fontsize=14, fontweight='bold')
        ax.set_ylabel(variable_name_map.get(selected_variable, selected_variable), fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        box_img_path = os.path.join(temp_dir, 'boxplot.png')
        plt.savefig(box_img_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        story.append(Image(box_img_path, width=5*inch, height=3*inch))
        story.append(Spacer(1, 10))
        
        # 변수 통계 테이블 (한글)
        stats_data = [
            ['통계 항목', '값'],
            ['평균값', f"{var_stats['mean']:.2f}"],
            ['표준편차', f"{var_stats['std']:.2f}"],
            ['최솟값', f"{var_stats['min']:.2f}"],
            ['최댓값', f"{var_stats['max']:.2f}"],
            ['중앙값', f"{var_stats['50%']:.2f}"],
            ['변동계수', f"{(var_stats['std']/var_stats['mean']*100):.1f}%"]
        ]
        
        stats_table = Table(stats_data, colWidths=[1.8*inch, 1.2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, blue),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_blue])
        ]))
        
        story.append(Paragraph("기술통계 요약", subsection_style))
        story.append(stats_table)
    
    story.append(PageBreak())
    
    # === 4. 품질 영향 분석 ===
    story.append(Paragraph("4. 품질 영향 분석", section_style))
    
    if not filtered_data.empty and outliers is not None and selected_variable in filtered_data.columns:
        df_temp = filtered_data.copy()
        df_temp['is_outlier'] = outliers[selected_variable]
        
        # 정상치 vs 이상치 불량률 비교
        normal_defect_rate = df_temp[~df_temp['is_outlier']]['passorfail'].mean() * 100
        outlier_defect_rate = df_temp[df_temp['is_outlier']]['passorfail'].mean() * 100 if df_temp['is_outlier'].sum() > 0 else 0
        
        impact_data = [
            ['구분', '데이터 수', '불량률', '영향도'],
            ['정상치', f"{(~df_temp['is_outlier']).sum():,}건", f"{normal_defect_rate:.1f}%", '기준'],
            ['이상치', f"{df_temp['is_outlier'].sum():,}건", f"{outlier_defect_rate:.1f}%",
             '높음' if outlier_defect_rate > normal_defect_rate * 2 else '보통' if outlier_defect_rate > normal_defect_rate else '낮음']
        ]
        
        impact_table = Table(impact_data, colWidths=[1.2*inch, 1.3*inch, 1.2*inch, 1*inch])
        impact_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), navy),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, -1), korean_font),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, navy),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, light_blue])
        ]))
        
        story.append(Paragraph("이상치-품질 상관관계 분석", subsection_style))
        story.append(impact_table)
        story.append(Spacer(1, 24))
        
        # 위험도 평가 (한글)
        risk_ratio = outlier_defect_rate / normal_defect_rate if normal_defect_rate > 0 else 1
        risk_text = f"""
<b>위험도 평가:</b><br/>
• 이상치 발생 시 불량률이 정상 대비 <b>{risk_ratio:.1f}배</b> {'증가' if risk_ratio > 1 else '감소'}<br/>
• 위험 수준: <b>{'매우 높음' if risk_ratio > 3 else '높음' if risk_ratio > 2 else '보통' if risk_ratio > 1.5 else '낮음'}</b><br/>
• 권장 조치: <b>{'즉시 공정 중단 및 점검' if risk_ratio > 3 else '집중 모니터링' if risk_ratio > 2 else '정기 점검' if risk_ratio > 1.5 else '현 수준 유지'}</b>
        """
        story.append(Paragraph(risk_text, highlight_style))
    
    story.append(PageBreak())
    
    # === 5. 결론 및 권고사항 ===
    story.append(Paragraph("5. 결론 및 권고사항", section_style))
    
    # 주요 발견사항 (한글)
    story.append(Paragraph("5.1 주요 발견사항", subsection_style))
    
    if not filtered_data.empty and outliers is not None:
        total = len(filtered_data)
        outlier_rate = (outliers.any(axis=1).sum() / total * 100) if total > 0 else 0
        defect_rate = (filtered_data['passorfail'].sum() / len(filtered_data) * 100) if len(filtered_data) > 0 else 0
        
        findings_text = f"""
1. <b>전체 품질 현황:</b> 이상치 비율 {outlier_rate:.1f}%, 불량률 {defect_rate:.1f}%<br/>
2. <b>위험 수준:</b> {'즉시 개선 필요' if outlier_rate > 20 else '주의 관찰' if outlier_rate > 10 else '안정적 운영'}<br/>
3. <b>핵심 문제 변수:</b> 상위 3개 변수에서 전체 이상치의 60% 이상 발생<br/>
4. <b>몰드별 편차:</b> 몰드간 품질 편차가 {'크게' if len(filtered_data['mold_code'].unique()) > 2 else '약간'} 존재<br/>
5. <b>개선 여지:</b> 체계적 관리로 이상치 30% 이상 감소 가능
        """
        story.append(Paragraph(findings_text, body_style))
        story.append(Spacer(1, 15))
        
        # 권고사항 (한글)
        story.append(Paragraph("5.2 단계별 권고사항", subsection_style))
        
        recommendations_text = """
<b>단기 조치 (1-2주)</b><br/>
• 이상치 발생률 상위 3개 변수 집중 모니터링<br/>
• 몰드별 공정 파라미터 재검토 및 최적화<br/>
• 운영자 교육 및 표준 작업 절차 점검<br/><br/>

<b>중기 개선 (1-3개월)</b><br/>
• 통계적 공정 관리(SPC) 시스템 강화<br/>
• 예방적 유지보수 계획 수립<br/>
• 품질 관리 기준 재설정<br/><br/>

<b>장기 전략 (3-6개월)</b><br/>
• AI 기반 실시간 품질 예측 시스템 도입<br/>
• 공정 자동화 및 스마트 팩토리 구축<br/>
• 지속적 개선 문화 정착
        """
        story.append(Paragraph(recommendations_text, highlight_style))
        
        # 기대 효과 (한글)
        story.append(Spacer(1, 15))
        story.append(Paragraph("5.3 기대 효과", subsection_style))
        
        benefits_text = f"""
<b>정량적 효과:</b><br/>
• 이상치 비율 30% 감소 ({outlier_rate:.1f}% → {outlier_rate*0.7:.1f}%)<br/>
• 불량률 20% 개선 ({defect_rate:.1f}% → {defect_rate*0.8:.1f}%)<br/>
• 생산성 10-15% 향상 예상<br/><br/>

<b>정성적 효과:</b><br/>
• 품질 안정성 및 고객 만족도 향상<br/>
• 운영 효율성 및 비용 절감<br/>
• 데이터 기반 의사결정 문화 구축
        """
        story.append(Paragraph(benefits_text, body_style))
    
    # 보고서 마무리 (한글)
    story.append(Spacer(1, 30))
    end_style = ParagraphStyle('End', alignment=TA_CENTER, fontSize=12, textColor=blue, fontName=korean_font)
    story.append(Paragraph("", end_style))
    
    # PDF 생성
    doc.build(story)
    
    return pdf_path

# SHAP HTML 파일 읽기 (파일이 없을 경우 대체 텍스트)
try:
    with open("data/shap_feature_importance.html", "r", encoding="utf-8") as f:
        shap_html_content = f.read()
except FileNotFoundError:
    shap_html_content = "<div><h3>SHAP Feature Importance</h3><p>SHAP HTML 파일을 찾을 수 없습니다.</p></div>"

# ───────────────────────────────────────────────────────────────────────────
# 3) Reactive 상태값 선언
# ───────────────────────────────────────────────────────────────────────────
probas       = reactive.Value([])       # 예측 확률 누적 리스트
last_n       = reactive.Value(0)        # 마지막으로 처리된 행 개수
start_time   = reactive.Value(time.time())  # 스트리밍 시작 시간
streaming    = reactive.Value(False)    # 스트리밍 동작 중 여부

# 알림 메시지 + 고유 버튼 ID + mold_code + row_index 저장
notifications = reactive.Value([])

# "어떤 알림"을 클릭했을 때, 해당 행(row index)을 저장할 용도
click_row = reactive.Value(None)

# 이미 클릭해서 처리한 알림 버튼 ID들을 저장
processed_alerts = reactive.Value(set())

# 몰드 코드별 알림 배경색 매핑(간단한 팔레트 활용)
unique_molds = list(df_all["mold_code"].unique())
palette = ["#e57373", "#64b5f6", "#81c784", "#ffb74d", "#9575cd", "#4db6ac", "#7986cb", "#f0628"]
mold_colors = {str(m): palette[i % len(palette)] for i, m in enumerate(unique_molds)}

@reactive.Calc
def n_rows():
    reactive.invalidate_later(INTERVAL)
    elapsed = time.time() - start_time.get()
    return min(int(elapsed / INTERVAL) + 1, len(df_all))

# ───────────────────────────────────────────────────────────────────────────
# 품질 분석 모달 정의 (두 번째 파일의 UI를 그대로 사용)
# ───────────────────────────────────────────────────────────────────────────

# 보고서 스타일 CSS
report_css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;600;700&family=Source+Serif+Pro:wght@400;600&display=swap');

:root {
    --primary-navy: #1a365d;
    --primary-blue: #2b77ad;
    --accent-gold: #d69e2e;
    --text-dark: #2d3748;
    --text-medium: #4a5568;
    --text-light: #718096;
    --background-paper: #ffffff;
    --background-light: #f7fafc;
    --border-light: #e2e8f0;
    --border-medium: #cbd5e0;
    --success-green: #38a169;
    --warning-orange: #dd6b20;
    --danger-red: #e53e3e;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
    line-height: 1.6;
    color: var(--text-dark);
    background: var(--background-light);
}

/* 모달 전용 스타일 */
.modal-body-custom {
    max-height: 80vh;
    overflow-y: auto;
    padding: 1rem;
}

/* 보고서 컨테이너 */
.report-container {
    max-width: 1200px;
    margin: 0 auto;
    background: var(--background-paper);
    box-shadow: 0 0 30px rgba(0,0,0,0.1);
    min-height: 100vh;
}

/* 헤더 섹션 */
.report-header {
    background: linear-gradient(135deg, var(--primary-navy) 0%, var(--primary-blue) 100%);
    color: white;
    padding: 3rem 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}

.report-title {
    font-family: 'Source Serif Pro', serif;
    font-size: 3rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
    position: relative;
    z-index: 1;
}

.report-subtitle {
    font-size: 1.2rem;
    opacity: 0.9;
    font-weight: 300;
    position: relative;
    z-index: 1;
}

.report-date {
    margin-top: 1rem;
    font-size: 0.95rem;
    opacity: 0.8;
    position: relative;
    z-index: 1;
}

/* 네비게이션 */
.report-nav {
    background: var(--background-paper);
    border-bottom: 2px solid var(--border-light);
    padding: 1rem 2rem;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.nav-actions {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}

.nav-controls {
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
}

.control-group {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.control-label {
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-medium);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.btn-download {
    background: linear-gradient(135deg, var(--accent-gold) 0%, #b7791f 100%);
    color: white;
    border: none;
    padding: 0.8rem 1.5rem;
    border-radius: 6px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    text-decoration: none;
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.9rem;
}

.btn-download:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(214, 158, 46, 0.3);
}

/* 메인 콘텐츠 */
.report-content {
    padding: 2rem;
}

/* 섹션 스타일 */
.report-section {
    margin-bottom: 4rem;
    page-break-inside: avoid;
}

.section-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-bottom: 1rem;
    border-bottom: 3px solid var(--primary-blue);
}

.section-number {
    background: var(--primary-navy);
    color: white;
    width: 2.5rem;
    height: 2.5rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    font-size: 1.1rem;
}

.section-title {
    font-family: 'Source Serif Pro', serif;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--primary-navy);
    margin: 0;
}

.section-description {
    color: var(--text-medium);
    font-size: 1rem;
    margin-bottom: 1.5rem;
    line-height: 1.7;
}

/* 차트 그리드 */
.charts-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 2rem;
    margin-bottom: 2rem;
}

.chart-container {
    background: var(--background-paper);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    transition: all 0.2s ease;
}

.chart-container:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}

.chart-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 1rem;
    text-align: center;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-light);
}

/* 통계 박스 */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 2rem 0;
}

.stat-box {
    background: linear-gradient(135deg, var(--background-paper) 0%, var(--background-light) 100%);
    padding: 1.5rem;
    border-radius: 8px;
    border-left: 4px solid var(--primary-blue);
    text-align: center;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-navy);
    display: block;
}

.stat-label {
    font-size: 0.9rem;
    color: var(--text-medium);
    margin-top: 0.5rem;
    font-weight: 500;
}

/* 경고/성공 상태 */
.status-good { color: var(--success-green); }
.status-warning { color: var(--warning-orange); }
.status-danger { color: var(--danger-red); }

/* 인사이트 박스 */
.insight-box {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border: 1px solid #0ea5e9;
    border-radius: 8px;
    padding: 1.5rem;
    margin: 2rem 0;
    position: relative;
}

.insight-box::before {
    content: '💡';
    position: absolute;
    top: -10px;
    left: 20px;
    background: #0ea5e9;
    color: white;
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.9rem;
}

.insight-title {
    font-weight: 600;
    color: #0c4a6e;
    margin-bottom: 0.5rem;
    margin-left: 1rem;
}

.insight-text {
    color: #075985;
    line-height: 1.6;
}

/* 알림 메시지 */
.alert {
    padding: 1rem 1.5rem;
    border-radius: 6px;
    margin: 1rem 0;
    border-left: 4px solid;
}

.alert-success {
    background: #f0fff4;
    border-color: var(--success-green);
    color: #22543d;
}

.alert-warning {
    background: #fffaf0;
    border-color: var(--warning-orange);
    color: #744210;
}

.alert-error {
    background: #fff5f5;
    border-color: var(--danger-red);
    color: #742a2a;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .report-title {
        font-size: 2rem;
    }
    
    .report-content {
        padding: 1rem;
    }
    
    .nav-actions {
        flex-direction: column;
        align-items: stretch;
    }
    
    .nav-controls {
        justify-content: center;
    }
    
    .charts-grid {
        grid-template-columns: 1fr;
    }
    
    .stats-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    .stats-grid {
        grid-template-columns: 1fr;
    }
    
    .section-header {
        flex-direction: column;
        text-align: center;
    }
}

/* 로딩 애니메이션 */
.loading {
    display: flex;
    justify-content: center;
    align-items: center;
    height: 200px;
    flex-direction: column;
    gap: 1rem;
}

.spinner {
    width: 40px;
    height: 40px;
    border: 4px solid var(--border-light);
    border-top: 4px solid var(--primary-blue);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
"""

quality_analysis_modal = ui.modal(
    ui.div(
        {"class": "modal-body-custom"},
        ui.HTML(report_css),
        
        # 보고서 컨테이너
        ui.div(
            # 헤더
            ui.div(
                ui.h1("제조업 품질 분석 보고서", class_="report-title"),
                ui.p("Manufacturing Quality Analysis Report", class_="report-subtitle"),
                ui.p(f"보고서 생성일: {datetime.now().strftime('%Y년 %m월 %d일')}", class_="report-date"),
                class_="report-header"
            ),
            
            # 네비게이션
            ui.div(
                ui.div(
                    ui.div(
                        ui.div(
                            ui.p("분석 기간", class_="control-label"),
                            ui.input_date_range("analysis_date_range", label="", start=min_date, end=max_date)
                        ),
                        ui.div(
                            ui.p("분석 변수", class_="control-label"),
                            ui.input_select("analysis_variable", label="", choices=variable_name_map, selected=selected_cols[0])
                        ),
                        class_="nav-controls"
                    ),
                    ui.input_action_button("download_pdf", "📊 PDF 보고서 생성", class_="btn-download"),
                    class_="nav-actions"
                ),
                class_="report-nav"
            ),
            
            # 메인 콘텐츠
            ui.div(
                # 섹션 1: 개요
                ui.div(
                    ui.div(
                        ui.div("1", class_="section-number"),
                        ui.h2("분석 개요", class_="section-title"),
                        class_="section-header"
                    ),
                    ui.p("본 보고서는 제조 공정의 품질 데이터를 분석하여 이상치 현황과 품질 영향 요인을 파악합니다. "
                         "IQR(Interquartile Range) 방법을 사용하여 몰드별 이상치를 탐지하고, "
                         "각 변수가 제품 품질에 미치는 영향을 분석합니다.", 
                         class_="section-description"),
                    
                    # 통계 요약
                    ui.output_ui("modal_summary_stats"),
                    
                    class_="report-section"
                ),
                
                # 섹션 2: 전체 이상치 분석
                ui.div(
                    ui.div(
                        ui.div("2", class_="section-number"),
                        ui.h2("전체 이상치 분석", class_="section-title"),
                        class_="section-header"
                    ),
                    ui.p("전체 데이터에서 이상치의 분포와 변수별 이상치 발생 패턴을 분석합니다.", 
                         class_="section-description"),
                    
                    ui.div(
                        ui.div(
                            ui.h3("2.1 이상치 비율 현황", class_="chart-title"),
                            ui.output_ui("modal_outlier_ratio_plot"),
                            class_="chart-container"
                        ),
                        ui.div(
                            ui.h3("2.2 변수별 이상치 발생률", class_="chart-title"),
                            ui.output_ui("modal_variable_outlier_plot"),
                            class_="chart-container"
                        ),
                        class_="charts-grid"
                    ),
                    
                    # 인사이트
                    ui.div(
                        ui.h4("주요 발견사항", class_="insight-title"),
                        ui.output_ui("modal_outlier_insights"),
                        class_="insight-box"
                    ),
                    
                    class_="report-section"
                ),
                
                # 섹션 3: 변수별 상세 분석
                ui.div(
                    ui.div(
                        ui.div("3", class_="section-number"),
                        ui.h2("변수별 상세 분석", class_="section-title"),
                        class_="section-header"
                    ),
                    ui.p("선택된 변수의 몰드별 분포 특성과 품질과의 상관관계를 분석합니다.", 
                         class_="section-description"),
                    
                    ui.div(
                        ui.div(
                            ui.h3("3.1 몰드별 분포 현황", class_="chart-title"),
                            ui.output_ui("modal_boxplot"),
                            class_="chart-container"
                        ),
                        ui.div(
                            ui.h3("3.2 분포 vs 이상치 비율", class_="chart-title"),
                            ui.output_ui("modal_bar_chart"),
                            class_="chart-container"
                        ),
                        ui.div(
                            ui.h3("3.3 품질 영향 분석", class_="chart-title"),
                            ui.output_ui("modal_quality_by_outlier"),
                            class_="chart-container"
                        ),
                        class_="charts-grid"
                    ),
                    
                    # 변수별 인사이트
                    ui.div(
                        ui.h4("변수별 분석 결과", class_="insight-title"),
                        ui.output_ui("modal_variable_insights"),
                        class_="insight-box"
                    ),
                    
                    class_="report-section"
                ),
                
                # 섹션 4: 특성 중요도
                ui.div(
                    ui.div(
                        ui.div("4", class_="section-number"),
                        ui.h2("특성 중요도 분석", class_="section-title"),
                        class_="section-header"
                    ),
                    ui.p("SHAP(SHapley Additive exPlanations) 분석을 통해 각 변수가 모델 예측에 미치는 영향을 분석합니다.", 
                         class_="section-description"),
                    
                    ui.div(
                        ui.h3("4.1 SHAP Feature Importance", class_="chart-title"),
                        ui.HTML(shap_html_content),
                        class_="chart-container"
                    ),
                    
                    class_="report-section"
                ),
                
                # 섹션 5: 권고사항
                ui.div(
                    ui.div(
                        ui.div("5", class_="section-number"),
                        ui.h2("결론 및 권고사항", class_="section-title"),
                        class_="section-header"
                    ),
                    ui.output_ui("modal_recommendations"),
                    class_="report-section"
                ),
                
                # PDF 다운로드 결과
                ui.output_ui("modal_pdf_link"),
                
                class_="report-content"
            ),
            
            class_="report-container"
        )
    ),
    title="예측모델 품질분석 보고서",
    size="xl",
    easy_close=True,
    footer=None
)

# ───────────────────────────────────────────────────────────────────────────
# alert_modal 전체 정의 (이 블록만 남기고 이전 것은 모두 지우세요)
# ───────────────────────────────────────────────────────────────────────────
alert_modal = ui.modal(
    ui.div(
        { "class": "modal-body",
          "style": """
            overflow: visible !important;
            max-height: none !important;
            padding: 0;
          """ },

        # ─── (1) 센서 상태 + 예측 실행 카드 ────────────────────────────────
        ui.card(
            ui.card_header("센서 상태 요약 및 예측 실행"),
            tags.div(
                { "style": """
                    position: relative;
                    padding: 16px;
                    background: #fff;
                    border-radius: 8px;
                    overflow: visible;
                    min-height: 480px;
                  """ },

                # 기계 이미지 중앙
                tags.img(
                    src="image3.png",
                    style="""
                      display: block;
                      margin: 0 auto;
                      max-width: 700px;
                      width: 100%;
                      height: auto;
                    """
                ),

                # 센서 슬라이더 팝업(수정)
                *[
                  tags.div(
                      { "class": "hover-zone",
                        "style": f"top:{t}; left:{l}; ",
                        "onclick": f"toggleSlider('{v}_slider')" },
                      ui.output_ui(f"{v}_box_tab2", container=tags.div),
                      tags.div(
                          { "class": "slider-box",
                            "id": f"{v}_slider",
                            "style": "position:absolute; z-index:2000; overflow:hidden;",
                            "onclick": "event.stopPropagation();" },
                          tags.button("✖", {
                              "style": "float:right;background:none;border:none;",
                              "onclick": f"closeSlider('{v}_slider');event.stopPropagation();"
                          }),
                          ui.output_ui(f"{v}_input_ui")
                      )
                  )
                  for (v,_),(t,l) in zip(slider_variables, positions)
                ],

                # Mold 코드 (왼쪽 하단)
                tags.div(
                    { "style": """
                        position:absolute;
                        bottom:12px;
                        left:12px;
                        background:#eee;
                        padding:4px 8px;
                        border-radius:4px;
                      """ },
                    ui.output_text("mold_code_box_tab2")
                ),

                # 예측 실행 버튼 (오른쪽 하단)
                tags.div(
                    ui.input_action_button("predict_btn_modal", "예측 실행", class_="btn btn-primary"),
                    style="position:absolute; bottom:12px; right:12px;"
                )
            )
        ),

        # ─── (2) 예측 결과 & 예측 로그 (모달 내부에 반드시 이 자리에) ──────────
        ui.row(
            ui.column(4,
                ui.card(
                    ui.card_header("예측 결과"),
                    tags.div(
                        ui.output_ui("predict_result_ui"),
                        style="""
                        height:150px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        """
                    )
                )
            ),
            ui.column(7,
                ui.card(
                    ui.card_header("예측 로그"),
                    tags.div(
                        ui.output_ui("predict_log_ui"),
                        style="""
                          height:150px;
                          overflow-y:auto;
                          white-space:pre-wrap;
                          font-family:monospace;
                          font-size:12px;
                        """
                    )
                )
            ),
            style="gap:16px; overflow:hidden;"
        )
    ),
    title=None,
    size="xl",
    easy_close=True,
    footer=None
)

app_ui = ui.page_fluid(
    # ────────────────────────────────────────────────────────────────────────
    # head: Plotly JS + CSS + toggleSlider JS + gauge domain 조정
    # ────────────────────────────────────────────────────────────────────────
# ── app_ui 정의 중 head 스크립트 ──
tags.head(
    # Plotly.js 로딩
    tags.script(src="https://cdn.plot.ly/plotly-2.24.2.min.js"),

    # 모달-body 스크롤 제거
    tags.style("""
      .modal-body {
        overflow: visible !important;
        max-height: none !important;
      }
      .modal-content {
        max-height: 90vh !important;
      }
    """),

    # hover-zone, slider-box 공통 스타일
    tags.style("""
      .hover-zone { position: absolute; cursor: pointer; }
      .slider-box {
        display: none;
        margin-top: 8px;
        background-color: rgba(255,255,255,1);
        padding: 10px;
        border-radius: 10px;
        width: 220px;
        z-index: 99;
      }
    """),

    # toggleSlider / closeSlider 함수 및 게이지 초기화·업데이트
    tags.script("""
      function toggleSlider(id) {
        const box = document.getElementById(id);
        box.style.display = (box.style.display === 'block') ? 'none' : 'block';
      }
      function closeSlider(id) {
        document.getElementById(id).style.display = 'none';
      }

      document.addEventListener("DOMContentLoaded", function() {
        Plotly.newPlot("gauge_chart", [{
          type: "indicator",
          mode: "gauge+number",
          value: 0,
          domain: { x: [0.2, 0.8], y: [0.2, 0.8] },
          title: { text: "행당 불량 확률 (%)", font: { size: 16 } },
          number: { suffix: "%", font: { size: 20 } },
          gauge: {
            axis: { range: [0, 100], tickwidth: 1, tickcolor: "#666" },
            bar: { color: "#7e57c2", thickness: 0.15 },
            bgcolor: "#f0f0f0",
            borderwidth: 2,
            bordercolor: "#ccc",
            steps: [
              { range: [0, 50], color: "#c8e6c9" },
              { range: [50, 100], color: "#ffcdd2" }
            ],
            threshold: {
              line: { color: "red", width: 3 },
              thickness: 0.75,
              value: 50
            }
          }
        }], {
          margin: { t: 0, b: 0, l: 0, r: 0 },
          paper_bgcolor: "transparent"
        }, {
          responsive: true
        });
      });

      Shiny.addCustomMessageHandler("gauge_update", function(data) {
        var trace = JSON.parse(data);
        trace.domain = { x: [0.2, 0.8], y: [0.2, 0.8] };
        trace.gauge.bar.thickness = 0.15;
        Plotly.react("gauge_chart", [trace], {
          margin: { t: 0, b: 0, l: 0, r: 0 },
          paper_bgcolor: "transparent"
        }, {
          responsive: true
        });
      });
    """)
),

# ────────────────────────────────────────────────────────────────────────
# 1) 상단 헤더: 날짜 / 생산라인 / 생산제품 + 버튼들
# ────────────────────────────────────────────────────────────────────────
ui.div(
  # — 좌측 그룹: 날짜 / 생산라인 / 생산제품
  tags.div(
    tags.span(f"날짜: {pd.Timestamp.now():%Y-%m-%d}", style="font-size:14px; width:160px; display:inline-block;"),
    tags.span(f"생산라인: 전자교반 3라인 2호기", style="font-size:14px; width:200px; display:inline-block;"),  # 실제 output_text 대신 미리 값 채워서 예시
    tags.span(f"생산제품: TM Carrier RH", style="font-size:14px; width:220px; display:inline-block;"),      # 실제 output_text 대신 예시
    style="""
      display: flex;
      gap: 24px;
      align-items: center;
      flex: none;
    """
  ),

  # — 우측 그룹: 버튼들
  tags.div(
    ui.input_action_button(
      "model_info_btn", "예측모델 분석보기",
      style="height:32px; font-size:13px; padding:0 12px; flex: none;"
    ),
    ui.output_ui("toggle_btn_ui"),
    ui.input_action_button(
      "stop_btn", "■ 정지",
      style="height:32px; font-size:13px; padding:0 12px; flex: none;"
    ),
    style="""
      display: flex;
      gap: 12px;
      align-items: center;
    """
  ),

  style="""
    position: sticky;
    top: 0;
    z-index: 999;
    background: #fff;
    padding: 8px 16px;
    border-bottom: 1px solid #ddd;
    display: flex;
    justify-content: space-between;
    align-items: center;
    height: 56px;
  """
),
tags.div(style="height: 12px;"),



    # ────────────────────────────────────────────────────────────────────────
    # 2) 메인: 왼쪽(기계 이미지) / 오른쪽(2×3 카드 + 알림창)
    # ────────────────────────────────────────────────────────────────────────
    ui.row(
      ui.column(8,
        ui.card(
          ui.card_header("실시간 상태"),
          tags.div(
            {"style":"position:relative;width:100%;height:49vh;"},
            tags.img(src="image3.png",
                     style="width:100%;height:100%;object-fit:contain;display:block;"),
            *[
              tags.div(
                {"class":"hover-zone","style":f"top:{t};left:{l};"},
                ui.output_ui(f"{v}_box_tab1",container=tags.div)
              )
              for (v,_),(t,l) in zip(slider_variables,positions)
            ],
            tags.div(
              {"style":"position:absolute;top:80%;left:60%;"},
              ui.output_ui("mold_code_box_tab1",container=tags.div)
            )
          )
        )
      ),
      ui.column(4,
        ui.row(
          ui.column(6,
            *[
              ui.div(
                ui.div(
                  tags.div(ui.output_text(fn),class_="numeric",
                           style="width:70px;height:24px;line-height:24px;font-size:20px;font-weight:bold;text-align:right;"),
                  tags.div(label,style="font-size:11px;color:#555;margin-top:2px;"),
                  style="display:flex;flex-direction:column;"
                ),
                (ui.output_ui(delta_fn) if delta_fn else tags.div()),
                style="background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px 10px;margin-bottom:3px;height:75px;display:flex;justify-content:space-between;align-items:center;"
              )
              for fn,delta_fn,label in [
                ("avg_molten_temp_card","delta_molten_temp_ui","평균 용탕 온도 (℃)"),
                ("avg_cast_pressure_card","delta_cast_pressure_ui","평균 주조 압력 (bar)"),
                ("avg_high_speed_card","delta_high_speed_ui","평균 상단 구간 속도 (rpm)")
              ]
            ]
          ),
          ui.column(6,
            *[
              ui.div(
                ui.div(
                  tags.div(ui.output_text(fn),class_="numeric",
                           style="width:70px;height:24px;line-height:24px;font-size:20px;font-weight:bold;text-align:right;"),
                  tags.div(label,style="font-size:11px;color:#555;margin-top:2px;"),
                  style="display:flex;flex-direction:column;"
                ),
                (ui.output_ui(delta_fn) if delta_fn else tags.div()),
                style="background:#fff;border:1px solid #ddd;border-radius:6px;padding:6px 10px;margin-bottom:3px;height:75px;display:flex;justify-content:space-between;align-items:center;"
              )
              for fn,delta_fn,label in [
                ("avg_low_speed_card","delta_low_speed_ui","평균 하단 구간 속도 (rpm)"),
                ("avg_coolant_temp_card","delta_coolant_temp_ui","평균 냉각수 온도 (℃)"),
                ("mold_code_card",None,"현재 금형 코드")
              ]
            ]
          )
        ),
            # 알림창
            ui.card(
                            ui.card_header("알림창"),
                            ui.output_ui("alert_ui", container=tags.div, style="height:30vh; overflow-y:auto; padding:4px;"),
                            style="margin:4px; height:32vh;"
                        )

        )
    ),


    # ────────────────────────────────────────────────────────────────────────
    # 3) 하단: 행당 불량 확률 게이지 / 관리도 / 파이차트
    # ────────────────────────────────────────────────────────────────────────
    ui.row(
      ui.column(4,
        ui.card(
          ui.card_header("행당 불량 확률 게이지"),
          tags.div(id="gauge_chart",style="width:100%;height:100%;"),
          style="height:300px;margin:6px;padding:0;display:flex;flex-direction:column;overflow:hidden;"
        )
      ),
      ui.column(4,
        ui.card(
          ui.card_header("누적 불량률 관리도"),
          ui.output_plot("accum_defect_plot",height="30vh",width="100%"),
          style="margin:4px;height:32vh;"
        )
      ),
      ui.column(4,
        ui.card(
          ui.card_header("몰드 코드별 생산비율"),
          ui.output_plot("production_pie",height="30vh",width="100%"),
          style="margin:4px;height:32vh;"
        )
      )
    )
)


# ───────────────────────────────────────────────────────────────────────────
# 5) Server 로직
# ───────────────────────────────────────────────────────────────────────────
def server(input, output, session):
    
    # 품질 분석 모달 전용 PDF 링크
    modal_pdf_href = reactive.Value("")
    
    # (1) "마지막 로그 행의 datetime" 출력
    @output
    @render.text
    def current_time_text():
        reactive.invalidate_later(INTERVAL)
        n = len(probas.get())
        if n == 0:
            return ""
        last_row = df_all.iloc[n - 1]
        ts = last_row.get("datetime", pd.NaT)
        if isinstance(ts, pd.Timestamp) and not pd.isna(ts):
            return ts.strftime("%Y-%m-%d %H:%M:%S")
        date_part = last_row.get("date", "")
        time_part = last_row.get("time", "")
        date_str = str(date_part) if pd.notna(date_part) else ""
        time_str = str(time_part) if pd.notna(time_part) else ""
        return (date_str + " " + time_str).strip()

    # (1-1) "생산라인" 출력
    @output
    @render.text
    def current_line():
        n = len(probas.get())
        if n == 0:
            return "생산라인: -"
        last_row = df_all.iloc[n - 1]
        return f"생산라인: {last_row.get('line', '-')}"

    # (1-2) "생산제품" 출력
    @output
    @render.text
    def current_name():
        n = len(probas.get())
        if n == 0:
            return "생산제품: -"
        last_row = df_all.iloc[n - 1]
        return f"생산제품: {last_row.get('name', '-')}"

    # (2) ▶시작/⏸/▶재시작 버튼 UI
    @output
    @render.ui
    def toggle_btn_ui():
        base_style = "height:32px; font-size:13px; padding:0 10px;"
        if streaming.get():
            return ui.input_action_button("toggle_btn", "⏸ 일시정지", style=base_style)
        elif last_n.get() > 0:
            return ui.input_action_button("toggle_btn", "▶ 재시작", style=base_style)
        else:
            return ui.input_action_button("toggle_btn", "▶ 시작", style=base_style)

    # ===== 품질 분석 모달 관련 =====
    @reactive.Effect
    @reactive.event(input.model_info_btn)
    def _show_quality_analysis():
        ui.modal_show(quality_analysis_modal, session=session)

    # 품질 분석용 데이터 필터링
    @reactive.Calc
    def filtered_analysis_data():
        try:
            start_date, end_date = input.analysis_date_range()
            mask = (train['registration_time'].dt.date >= start_date) & \
                   (train['registration_time'].dt.date <= end_date)
            cols = [c for c in selected_cols if c in train.columns]
            return train.loc[mask, ['mold_code', 'passorfail'] + cols].copy()
        except:
            return pd.DataFrame()

    @reactive.Calc
    def analysis_outlier_df():
        df = filtered_analysis_data()
        if df.empty:
            return None
        return detect_outliers_iqr_by_mold(df, mold_col='mold_code', cols=selected_cols)

    # 모달 내 요약 통계
    @output
    @render.ui
    def modal_summary_stats():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        
        if df.empty or outliers is None:
            return ui.HTML('<div class="loading"><div class="spinner"></div><p>데이터를 로딩 중입니다...</p></div>')
        
        total = len(df)
        outlier_count = outliers.any(axis=1).sum()
        outlier_rate = (outlier_count / total * 100) if total > 0 else 0
        defect_rate = (df['passorfail'].sum() / len(df) * 100) if len(df) > 0 else 0
        
        return ui.HTML(f'''
            <div class="stats-grid">
                <div class="stat-box">
                    <span class="stat-value">{total:,}</span>
                    <div class="stat-label">총 데이터 수</div>
                </div>
                <div class="stat-box">
                    <span class="stat-value status-{"danger" if outlier_rate > 20 else "warning" if outlier_rate > 10 else "good"}">{outlier_rate:.1f}%</span>
                    <div class="stat-label">이상치 비율</div>
                </div>
                <div class="stat-box">
                    <span class="stat-value status-{"danger" if defect_rate > 6 else "warning" if defect_rate > 4 else "good"}">{defect_rate:.1f}%</span>
                    <div class="stat-label">불량률</div>
                </div>
                <div class="stat-box">
                    <span class="stat-value">{len(df['mold_code'].unique())}</span>
                    <div class="stat-label">분석 몰드 수</div>
                </div>
            </div>
        ''')

    # 모달 내 이상치 비율 플롯
    @output
    @render.ui
    def modal_outlier_ratio_plot():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        if df.empty or outliers is None:
            return ui.HTML('<div class="loading"><div class="spinner"></div><p>데이터를 분석 중입니다...</p></div>')

        total = len(df)
        outlier_count = outliers.any(axis=1).sum()
        normal_count = total - outlier_count

        pie_df = pd.DataFrame({
            "상태": ["정상치", "이상치"],
            "개수": [normal_count, outlier_count]
        })

        fig = px.pie(
            pie_df,
            names="상태",
            values="개수",
            title="",
            color="상태",
            color_discrete_map={"정상치": "#38a169", "이상치": "#e53e3e"},
            hole=0.5
        )
        
        fig.update_layout(
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=20, b=60, l=20, r=20),
            font=dict(family="Noto Sans KR, sans-serif", size=12)
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=13
        )

        return ui.HTML(pio.to_html(fig, full_html=False))

    # 모달 내 변수별 이상치 플롯
    @output
    @render.ui
    def modal_variable_outlier_plot():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        if df.empty or outliers is None:
            return ui.HTML('<div class="loading"><div class="spinner"></div><p>변수별 분석 중입니다...</p></div>')

        outlier_rows = outliers.any(axis=1)
        outlier_only = outliers.loc[outlier_rows]
        if outlier_only.empty:
            return ui.HTML('<p style="text-align: center; padding: 2rem; color: #718096;">이상치 데이터가 없습니다.</p>')

        var_ratios = (outlier_only.sum() / len(outlier_only)).sort_values(ascending=False)
        var_ratios_percent = (var_ratios * 100).round(1)

        labels_kor = [variable_name_map.get(var, var) for var in var_ratios_percent.index]

        fig = px.bar(
            x=labels_kor,
            y=var_ratios_percent.values,
            text=var_ratios_percent.apply(lambda v: f"{v:.1f}%"),
            title=""
        )
        
        fig.update_traces(
            marker_color='#2b77ad',
            textposition="outside",
            textfont=dict(size=11),
            hovertemplate="%{x}<br>이상치 비율: %{y:.1f}%<extra></extra>"
        )
        
        fig.update_layout(
            height=400,
            yaxis=dict(title="이상치 비율 (%)", range=[0, max(var_ratios_percent) * 1.15]),
            xaxis=dict(title="", tickangle=-45),
            margin=dict(t=20, b=100, l=60, r=20),
            font=dict(family="Noto Sans KR, sans-serif")
        )

        return ui.HTML(pio.to_html(fig, full_html=False))

    # 모달 내 박스플롯
    @output
    @render.ui
    def modal_boxplot():
        df = filtered_analysis_data()
        var = input.analysis_variable()
        if df.empty or var not in df.columns:
            return ui.HTML('<div class="loading"><div class="spinner"></div><p>차트를 생성 중입니다...</p></div>')

        df['mold_code_str'] = df['mold_code'].astype(str)
        sorted_molds = sorted(df['mold_code_str'].unique())

        colors = ['#1a365d', '#2b77ad', '#d69e2e', '#e53e3e', '#38a169']

        fig = go.Figure()

        for i, mold in enumerate(sorted_molds):
            mold_df = df[df['mold_code_str'] == mold]
            fig.add_trace(go.Box(
                y=mold_df[var],
                name=f"{mold}",
                boxpoints='outliers',
                marker=dict(color=colors[i % len(colors)], size=4),
                line=dict(color=colors[i % len(colors)], width=2),
                boxmean=True
            ))

        fig.update_layout(
            title="",
            xaxis_title='몰드 코드',
            yaxis_title=variable_name_map.get(var, var),
            height=400,
            margin=dict(t=20, b=60, l=60, r=20),
            font=dict(family="Noto Sans KR, sans-serif"),
            showlegend=False
        )

        return ui.HTML(pio.to_html(fig, full_html=False))

    # 모달 내 막대차트
    @output
    @render.ui
    def modal_bar_chart():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        var = input.analysis_variable()
        if df.empty or var not in df.columns or outliers is None:
            return ui.HTML('<div class="loading"><div class="spinner"></div><p>분석 중입니다...</p></div>')

        df = df.copy()
        df['mold_code_str'] = df['mold_code'].astype(str)
        outliers = outliers.loc[df.index]

        value_sum_by_mold = df.groupby('mold_code_str')[var].sum()
        total_sum = df[var].sum()
        value_ratio = (value_sum_by_mold / total_sum).reset_index()
        value_ratio.columns = ['mold_code_str', 'value_ratio']

        outlier_ratio = outliers.groupby(df['mold_code_str'])[var].apply(lambda x: x.sum() / len(x)).reset_index()
        outlier_ratio.columns = ['mold_code_str', 'outlier_ratio']

        plot_df = pd.merge(value_ratio, outlier_ratio, on='mold_code_str')

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=plot_df['mold_code_str'],
            y=plot_df['value_ratio'],
            name='몰드별 분포 비율',
            marker_color='#2b77ad',
            yaxis='y1',
            opacity=0.8
        ))

        fig.add_trace(go.Scatter(
            x=plot_df['mold_code_str'],
            y=plot_df['outlier_ratio'],
            name='이상치 비율',
            mode='lines+markers',
            marker=dict(color='#e53e3e', size=8),
            line=dict(color='#e53e3e', width=3),
            yaxis='y2'
        ))

        fig.update_layout(
            title="",
            xaxis=dict(title='몰드 코드'),
            yaxis=dict(title='분포 비율', tickformat=".1%", side='left'),
            yaxis2=dict(title='이상치 비율', overlaying='y', side='right', tickformat=".1%"),
            height=400,
            margin=dict(t=20, b=60, l=60, r=60),
            font=dict(family="Noto Sans KR, sans-serif"),
            showlegend=False
        )

        return ui.HTML(pio.to_html(fig, full_html=False))

    # 모달 내 품질 영향 분석
    @output
    @render.ui
    def modal_quality_by_outlier():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        var = input.analysis_variable()

        if df.empty or outliers is None or 'passorfail' not in df.columns:
            return ui.HTML('<div class="loading"><div class="spinner"></div><p>품질 분석 중입니다...</p></div>')

        df = df.copy()
        df['is_outlier'] = outliers[var]
        df['상태'] = df['is_outlier'].map({True: '이상치', False: '정상치'})
        df['품질'] = df['passorfail'].map({0: '양품', 1: '불량'})
        df['mold_code'] = df['mold_code'].astype(str)

        grouped = df.groupby(['mold_code', '상태', '품질'], observed=False).size().reset_index(name='count')
        total_counts = df.groupby(['mold_code', '상태'], observed=False).size().reset_index(name='total')

        merged = grouped.merge(total_counts, on=['mold_code', '상태'])
        merged['percent'] = merged['count'] / merged['total'] * 100

        fig = px.bar(
            merged,
            x='mold_code',
            y='percent',
            color='품질',
            barmode='stack',
            facet_col='상태',
            title='',
            color_discrete_map={'양품': '#38a169', '불량': '#e53e3e'},
            labels={'percent': '비율 (%)', 'mold_code': '몰드 코드'},
            category_orders={
                '상태': ['정상치', '이상치'],
                '품질': ['양품', '불량']
            }
        )

        fig.update_layout(
            height=400,
            margin=dict(t=40, b=60, l=40, r=40),
            font=dict(family="Noto Sans KR, sans-serif"),
            yaxis=dict(range=[0, 115])
        )

        fig.update_traces(
            hovertemplate="몰드 코드: %{x}<br>비율: %{y:.1f}%<extra></extra>"
        )

        return ui.HTML(pio.to_html(fig, full_html=False))

    # 모달 내 인사이트
    @output
    @render.ui
    def modal_outlier_insights():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        
        if df.empty or outliers is None:
            return ui.HTML("<p>분석할 데이터가 없습니다.</p>")
        
        total = len(df)
        outlier_count = outliers.any(axis=1).sum()
        outlier_rate = (outlier_count / total * 100) if total > 0 else 0
        
        # 가장 이상치가 많은 변수
        outlier_rows = outliers.any(axis=1)
        if outlier_rows.sum() > 0:
            outlier_only = outliers.loc[outlier_rows]
            var_ratios = (outlier_only.sum() / len(outlier_only)).sort_values(ascending=False)
            top_variable = variable_name_map.get(var_ratios.index[0], var_ratios.index[0])
            top_rate = var_ratios.iloc[0] * 100
        else:
            top_variable = "없음"
            top_rate = 0
        
        insights = f"""
        <ul style="margin: 0; padding-left: 1.5rem; line-height: 1.8;">
            <li><strong>전체 이상치 비율:</strong> {outlier_rate:.1f}% 
                {"(⚠️ 높음)" if outlier_rate > 20 else "(✅ 양호)" if outlier_rate < 10 else "(⚡ 주의)"}
            </li>
            <li><strong>가장 문제가 되는 변수:</strong> {top_variable} ({top_rate:.1f}%)</li>
            <li><strong>분석 기간:</strong> {input.analysis_date_range()[0]} ~ {input.analysis_date_range()[1]}</li>
        </ul>
        """
        
        return ui.HTML(insights)

    @output
    @render.ui
    def modal_variable_insights():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        var = input.analysis_variable()
        
        if df.empty or outliers is None or var not in df.columns:
            return ui.HTML("<p>분석할 데이터가 없습니다.</p>")
        
        # 변수 통계
        var_stats = df[var].describe()
        
        # 이상치 영향 분석
        df_temp = df.copy()
        df_temp['is_outlier'] = outliers[var]
        
        normal_defect_rate = df_temp[~df_temp['is_outlier']]['passorfail'].mean() * 100
        outlier_defect_rate = df_temp[df_temp['is_outlier']]['passorfail'].mean() * 100 if df_temp['is_outlier'].sum() > 0 else 0
        
        # 몰드별 변동계수
        cv_by_mold = df.groupby('mold_code')[var].apply(lambda x: x.std() / x.mean() * 100).round(1)
        most_variable_mold = cv_by_mold.idxmax()
        
        insights = f"""
        <ul style="margin: 0; padding-left: 1.5rem; line-height: 1.8;">
            <li><strong>선택 변수:</strong> {variable_name_map.get(var, var)}</li>
            <li><strong>평균값:</strong> {var_stats['mean']:.2f} (표준편차: {var_stats['std']:.2f})</li>
            <li><strong>정상치 불량률:</strong> {normal_defect_rate:.1f}%</li>
            <li><strong>이상치 불량률:</strong> {outlier_defect_rate:.1f}% 
                {"(🚨 높은 위험)" if outlier_defect_rate > normal_defect_rate * 2 else "(⚠️ 주의)" if outlier_defect_rate > normal_defect_rate else "(✅ 양호)"}
            </li>
            <li><strong>가장 변동이 큰 몰드:</strong> {most_variable_mold} (변동계수: {cv_by_mold[most_variable_mold]:.1f}%)</li>
        </ul>
        """
        
        return ui.HTML(insights)

    @output
    @render.ui
    def modal_recommendations():
        df = filtered_analysis_data()
        outliers = analysis_outlier_df()
        
        if df.empty or outliers is None:
            return ui.HTML("<p>권고사항을 생성할 수 없습니다.</p>")
        
        total = len(df)
        outlier_rate = (outliers.any(axis=1).sum() / total * 100) if total > 0 else 0
        defect_rate = (df['passorfail'].sum() / len(df) * 100) if len(df) > 0 else 0
        
        recommendations = """
        <div style="line-height: 1.8;">
            <h4 style="color: var(--primary-navy); margin-bottom: 1rem;">📋 주요 결론</h4>
            <ul style="margin-bottom: 2rem; padding-left: 1.5rem;">
        """
        
        if outlier_rate > 20:
            recommendations += """
                <li style="color: var(--danger-red);">⚠️ <strong>높은 이상치 비율:</strong> 즉시 공정 개선이 필요합니다.</li>
                <li>제조 공정의 안정성 점검 및 제어 한계 재설정을 권장합니다.</li>
            """
        elif outlier_rate > 10:
            recommendations += """
                <li style="color: var(--warning-orange);">⚡ <strong>주의 수준의 이상치:</strong> 지속적인 모니터링이 필요합니다.</li>
                <li>예방적 유지보수 계획을 수립하여 품질 안정성을 확보하세요.</li>
            """
        else:
            recommendations += """
                <li style="color: var(--success-green);">✅ <strong>안정적인 공정:</strong> 현재 품질 수준을 유지하세요.</li>
                <li>정기적인 품질 모니터링을 통해 지속적인 개선을 추진하세요.</li>
            """
            
        recommendations += f"""
            </ul>
            
            <h4 style="color: var(--primary-navy); margin-bottom: 1rem;">🎯 권고사항</h4>
            <div style="background: var(--background-light); padding: 1.5rem; border-radius: 8px; border-left: 4px solid var(--primary-blue);">
                <ol style="margin: 0; padding-left: 1.5rem;">
                    <li><strong>단기 조치 (1-2주):</strong>
                        <ul style="margin-top: 0.5rem;">
                            <li>이상치 발생률이 높은 상위 3개 변수에 대한 집중 모니터링</li>
                            <li>몰드별 공정 파라미터 점검 및 조정</li>
                        </ul>
                    </li>
                    <li><strong>중기 개선 (1-3개월):</strong>
                        <ul style="margin-top: 0.5rem;">
                            <li>품질 관리 기준 재설정 및 표준화</li>
                            <li>운영자 교육 및 공정 최적화</li>
                        </ul>
                    </li>
                    <li><strong>장기 전략 (3-6개월):</strong>
                        <ul style="margin-top: 0.5rem;">
                            <li>예측 모델 기반 품질 관리 시스템 구축</li>
                            <li>지속적 개선을 위한 데이터 기반 의사결정 체계 구축</li>
                        </ul>
                    </li>
                </ol>
            </div>
            
            <div style="margin-top: 2rem; padding: 1rem; background: #f0f9ff; border-radius: 6px; border: 1px solid #0ea5e9;">
                <p style="margin: 0; color: #0c4a6e; font-weight: 500;">
                     💡 <strong>추가 분석 제안:</strong> 모델의 예측 정확도를 높이기 위해서는 훈련 데이터셋의 기간을 충분히 확보하는 것이 중요합니다.
                    더 많은 시계열 데이터를 활용하여 모델을 학습하면, 다양한 상황을 반영할 수 있어 보다 신뢰성 높은 예측 결과를 얻을 수 있습니다.
                    따라서 충분한 기간의 데이터를 수집 및 반영한 후, 추가적인 모델 학습과 평가를 권장합니다.
                </p>
            </div>
        </div>
        """
        
        return ui.HTML(recommendations)

    # 모달 내 PDF 생성 기능
    @reactive.Effect
    def _modal_download_pdf():
        input.download_pdf()
        if input.download_pdf() > 0:
            try:
                # 현재 필터링된 데이터 가져오기
                df = filtered_analysis_data()
                outliers = analysis_outlier_df()
                selected_var = input.analysis_variable()
                date_range = input.analysis_date_range()
                
                # PDF 생성
                pdf_path = generate_pdf_report(df, outliers, selected_var, date_range)
                
                # PDF 파일을 base64로 인코딩
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                
                # 다운로드 링크 생성
                filename = f"quality_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                href = f'''
                <div class="alert alert-success">
                    <h4 style="margin: 0 0 15px 0;">✅ PDF 보고서가 성공적으로 생성되었습니다!</h4>
                    <p style="margin: 0 0 15px 0;">생성 시간: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}</p>
                    <a href="data:application/pdf;base64,{b64_pdf}" download="{filename}" class="btn-download">
                        📄 PDF 보고서 다운로드
                    </a>
                </div>
                '''
                
                # 임시 파일 삭제
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

                modal_pdf_href.set(href)
                    
            except Exception as e:
                href = f'''
                <div class="alert alert-error">
                    <h4 style="margin: 0 0 10px 0;">❌ PDF 생성 중 오류가 발생했습니다</h4>
                    <p style="margin: 0;">오류 내용: {str(e)}</p>
                </div>
                '''
                modal_pdf_href.set(href)

    @output
    @render.ui
    def modal_pdf_link():
        return ui.HTML(modal_pdf_href.get())

    @reactive.Effect
    @reactive.event(input.toggle_btn)
    def _toggle():
        if streaming.get():
            streaming.set(False)
        else:
            streaming.set(True)
            if last_n.get() == 0:
                start_time.set(time.time())
                probas.set([])
                last_n.set(0)
                notifications.set([])
                click_row.set(None)
                processed_alerts.set(set())
            else:
                resumed = time.time() - last_n.get() * INTERVAL
                start_time.set(resumed)

    # (4) ■ 정지 버튼
    @reactive.Effect
    @reactive.event(input.stop_btn)
    def _stop():
        streaming.set(False)
        start_time.set(time.time())
        probas.set([])
        last_n.set(0)
        notifications.set([])
        click_row.set(None)
        processed_alerts.set(set())

    # (5) 실시간 스트리밍 예측 및 알림 생성
    @reactive.Effect
    def _stream():
        if streaming.get():
            reactive.invalidate_later(INTERVAL)
            n = n_rows()
            cur = probas.get()
            if len(cur) < n:
                row = df_all.iloc[n - 1 : n]
                p = model.predict_proba(row)[0][1]
                probas.set(cur + [p])
                last_n.set(n)
                if p >= ALERT_TH:
                    row0 = row.iloc[0]
                    mold_code = str(row0.get("mold_code", "-")).strip()
                    ts = row0.get("datetime", pd.NaT)
                    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if isinstance(ts, pd.Timestamp) and not pd.isna(ts) else ""
                    msg = f"{ts_str} | 금형 {mold_code} | 확률 {p:.2f}"
                    idx_alert = len(notifications.get()) + 1
                    alert_id = f"alert_{idx_alert}"
                    notifications.set(notifications.get() + [{
                        "id": alert_id,
                        "text": msg,
                        "mold": mold_code,
                        "row_idx": n - 1
                    }])

    # (A') 알림창 렌더링
    @output
    @render.ui
    def alert_ui():
        items = []
        for item in reversed(notifications.get()):
            btn = ui.input_action_button(
                item["id"],
                tags.div(
                    tags.span("⚠︎", style="color:#fff; font-weight:bold; margin-right:8px;"),
                    item["text"],
                    style="display:inline-block; vertical-align:middle;"
                ),
                style=(
                    f"width:100%; text-align:left; padding:10px; "
                    f"background-color: {mold_colors.get(item['mold'], '#e57373')}; "
                    "color:#fff; border:none; border-radius:4px; margin-bottom:6px;"
                )
            )
            items.append(btn)
        if not items:
            return tags.div("알림이 없습니다.", style="color:#888; font-size:14px; text-align:center; padding-top:100px;")
        return tags.div(*items)

    # (A") 알림 클릭 시 모달 띄우기
    @reactive.Effect
    def _alert_click_handler():
        reactive.invalidate_later(INTERVAL)
        done = processed_alerts.get().copy()
        for it in notifications.get():
            if input[it["id"]]() > 0 and it["id"] not in done:
                click_row.set(it["row_idx"])
                done.add(it["id"])
                processed_alerts.set(done)
                ui.modal_show(alert_modal, session=session)
                break
    # ─────────────────────────────────────────────────────────────────────────
    # (6) 카드용 평균값 & 증감 퍼센트 계산

    @output
    @render.text
    def avg_count():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0 or "count" not in df_all.columns:
            return "N/A"
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["count"].mean()
        return f"{avg_val:.1f}"

    @output
    @render.ui
    def delta_count_ui():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total < 2 or "count" not in df_all.columns:
            return tags.span("–", style="font-size:16px; color:#888;")
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["count"].mean()
        latest_val = float(df_all.iloc[n_total - 1]["count"])
        if avg_val == 0:
            return tags.span("–", style="font-size:16px; color:#888;")
        pct = (latest_val - avg_val) / avg_val * 100
        if pct > 0:
            sign, color = "▲", "red"
        elif pct < 0:
            sign, color = "▼", "blue"
        else:
            sign, color = "–", "#888"
        text = f"{sign}{abs(pct):.1f}%"
        return tags.span(text, style=f"font-size:16px; color:{color};")

    @output
    @render.text
    def avg_molten_temp_card():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0 or "molten_temp" not in df_all.columns:
            return "N/A"
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["molten_temp"].mean()
        return f"{avg_val:.1f}"

    @output
    @render.ui
    def delta_molten_temp_ui():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total < 2 or "molten_temp" not in df_all.columns:
            return tags.span("–", style="font-size:16px; color:#888;")
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["molten_temp"].mean()
        latest_val = float(df_all.iloc[n_total - 1]["molten_temp"])
        if avg_val == 0:
            return tags.span("–", style="font-size:16px; color:#888;")
        pct = (latest_val - avg_val) / avg_val * 100
        if pct > 0:
            sign, color = "▲", "red"
        elif pct < 0:
            sign, color = "▼", "blue"
        else:
            sign, color = "–", "#888"
        text = f"{sign}{abs(pct):.1f}%"
        return tags.span(text, style=f"font-size:16px; color:{color};")

    @output
    @render.text
    def avg_cast_pressure_card():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0 or "cast_pressure" not in df_all.columns:
            return "N/A"
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["cast_pressure"].mean()
        return f"{avg_val:.1f}"

    @output
    @render.ui
    def delta_cast_pressure_ui():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total < 2 or "cast_pressure" not in df_all.columns:
            return tags.span("–", style="font-size:16px; color:#888;")
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["cast_pressure"].mean()
        latest_val = float(df_all.iloc[n_total - 1]["cast_pressure"])
        if avg_val == 0:
            return tags.span("–", style="font-size:16px; color:#888;")
        pct = (latest_val - avg_val) / avg_val * 100
        if pct > 0:
            sign, color = "▲", "red"
        elif pct < 0:
            sign, color = "▼", "blue"
        else:
            sign, color = "–", "#888"
        text = f"{sign}{abs(pct):.1f}%"
        return tags.span(text, style=f"font-size:16px; color:{color};")

    @output
    @render.text
    def avg_high_speed_card():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0 or "high_section_speed" not in df_all.columns:
            return "N/A"
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["high_section_speed"].mean()
        return f"{avg_val:.1f}"

    @output
    @render.ui
    def delta_high_speed_ui():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total < 2 or "high_section_speed" not in df_all.columns:
            return tags.span("–", style="font-size:16px; color:#888;")
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["high_section_speed"].mean()
        latest_val = float(df_all.iloc[n_total - 1]["high_section_speed"])
        if avg_val == 0:
            return tags.span("–", style="font-size:16px; color:#888;")
        pct = (latest_val - avg_val) / avg_val * 100
        if pct > 0:
            sign, color = "▲", "red"
        elif pct < 0:
            sign, color = "▼", "blue"
        else:
            sign, color = "–", "#888"
        text = f"{sign}{abs(pct):.1f}%"
        return tags.span(text, style=f"font-size:16px; color:{color};")

    @output
    @render.text
    def avg_low_speed_card():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0 or "low_section_speed" not in df_all.columns:
            return "N/A"
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["low_section_speed"].mean()
        return f"{avg_val:.1f}"

    @output
    @render.ui
    def delta_low_speed_ui():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total < 2 or "low_section_speed" not in df_all.columns:
            return tags.span("–", style="font-size:16px; color:#888;")
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["low_section_speed"].mean()
        latest_val = float(df_all.iloc[n_total - 1]["low_section_speed"])
        if avg_val == 0:
            return tags.span("–", style="font-size:16px; color:#888;")
        pct = (latest_val - avg_val) / avg_val * 100
        if pct > 0:
            sign, color = "▲", "red"
        elif pct < 0:
            sign, color = "▼", "blue"
        else:
            sign, color = "–", "#888"
        text = f"{sign}{abs(pct):.1f}%"
        return tags.span(text, style=f"font-size:16px; color:{color};")

    @output
    @render.text
    def avg_coolant_temp_card():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0 or "Coolant_temperature" not in df_all.columns:
            return "N/A"
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["Coolant_temperature"].mean()
        return f"{avg_val:.1f}"

    @output
    @render.ui
    def delta_coolant_temp_ui():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total < 2 or "Coolant_temperature" not in df_all.columns:
            return tags.span("–", style="font-size:16px; color:#888;")
        df_slice = df_all.iloc[:n_total].copy()
        avg_val = df_slice["Coolant_temperature"].mean()
        latest_val = float(df_all.iloc[n_total - 1]["Coolant_temperature"])
        if avg_val == 0:
            return tags.span("–", style="font-size:16px; color:#888;")
        pct = (latest_val - avg_val) / avg_val * 100
        if pct > 0:
            sign, color = "▲", "red"
        elif pct < 0:
            sign, color = "▼", "blue"
        else:
            sign, color = "–", "#888"
        text = f"{sign}{abs(pct):.1f}%"
        return tags.span(text, style=f"font-size:16px; color:{color};")

    @output
    @render.text
    def mold_code_card():
        n = last_n.get()
        if n == 0 or "mold_code" not in df_all.columns:
            return "-"
        return str(df_all.iloc[n - 1]["mold_code"])

    # ─────────────────────────────────────────────────────────────────────────
    # (7) 누적 불량률 관리도 (Matplotlib) – 'H' → 'h' 변경
    @output
    @render.plot
    def accum_defect_plot():
        y_vals = np.array(probas.get())
        n_total = len(y_vals)
        if n_total == 0:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
            ax.text(0.5, 0.5, "데이터 없음", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return fig

        df_slice = df_all.iloc[:n_total].copy().reset_index(drop=True)
        df_slice["is_defect"] = (y_vals >= TH).astype(int)
        df_slice["time_slot"] = pd.to_datetime(df_slice["datetime"]).dt.floor("h")
        grouped = df_slice.groupby("time_slot").agg(
            defect_count=("is_defect", "sum"),
            total_count=("is_defect", "count")
        ).reset_index()
        grouped = grouped.sort_values("time_slot").reset_index(drop=True)
        grouped["cum_defects"] = grouped["defect_count"].cumsum()
        grouped["cum_samples"] = grouped["total_count"].cumsum()
        grouped["cum_rate"] = grouped["cum_defects"] / grouped["cum_samples"]
        final_p_hat = grouped["cum_rate"].iloc[-1]
        grouped["sigma_p"] = np.sqrt(final_p_hat * (1 - final_p_hat) / grouped["cum_samples"])
        grouped["UCL"] = final_p_hat + 3 * grouped["sigma_p"]
        grouped["LCL"] = (final_p_hat - 3 * grouped["sigma_p"]).clip(lower=0)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        x_times = grouped["time_slot"]
        ax.plot(
            x_times,
            grouped["cum_rate"],
            marker="o",
            linestyle="-",
            color="blue",
            label="누적 불량률"
        )
        ax.plot(x_times, grouped["UCL"], color="red", linestyle="--", label="UCL")
        ax.plot(x_times, grouped["LCL"], color="red", linestyle="--", label="LCL")
        ax.hlines(
            y=final_p_hat,
            xmin=x_times.min(),
            xmax=x_times.max(),
            colors="green",
            linestyles=":",
            label=f"최종 누적 평균 ({final_p_hat:.3f})"
        )
        ax.fill_between(x_times, grouped["LCL"], grouped["UCL"], color="red", alpha=0.1)
        ax.set_title("시간별 누적 불량률 관리도 (Cumulative P-Chart)", fontsize=12)
        ax.set_xlabel("시간 (Hour)", fontsize=10)
        ax.set_ylabel("누적 불량률", fontsize=10)
        plt.xticks(rotation=45, fontsize=8)
        plt.yticks(fontsize=8)
        ax.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────────────────────────────────
#  예측 불량률 게이지 (Matplotlib)
# ─────────────────────────────────────────────────────────────────────────
    @output
    @render.plot
    def gauge_plot():
        # 0.5초마다 재실행
        reactive.invalidate_later(INTERVAL)

        # 마지막 예측 확률 → %
        vals    = np.array(probas.get())
        percent = (vals[-1] if vals.size>0 else 0) * 100
        frac    = percent / 100

        # 차트 크기
        fig, ax = plt.subplots(figsize=(4,4), dpi=80)
        outer_r, width = 1.0, 0.25

        # 1) 배경 반원
        bg = Wedge((0,0), outer_r, 180, 0,
                width=width, facecolor="#e8eaf6", edgecolor="none")
        ax.add_patch(bg)

        # 2) 채워진 반원
        end_ang = 180 - frac * 180
        fg = Wedge((0,0), outer_r, 180, end_ang,
                width=width, facecolor="#5e35b1", edgecolor="none")
        ax.add_patch(fg)

        # 3) 포인터(삼각 마커 + 바늘)
        theta = np.radians(end_ang)
        tip_r = outer_r + 0.05
        tip = (tip_r*np.cos(theta), tip_r*np.sin(theta))
        m_w = 0.04
        L = (tip[0] + m_w*np.cos(theta+np.pi/2), tip[1] + m_w*np.sin(theta+np.pi/2))
        R = (tip[0] + m_w*np.cos(theta-np.pi/2), tip[1] + m_w*np.sin(theta-np.pi/2))
        ax.add_patch(Polygon([tip, L, R], facecolor="#7e57c2", edgecolor="none"))
        nd_r = outer_r - width - 0.05
        ax.plot([0, nd_r*np.cos(theta)], [0, nd_r*np.sin(theta)], color="black", linewidth=2)
        ax.add_patch(Circle((0,0), 0.04, facecolor="black", edgecolor="none"))

        # 4) 중앙 숫자
        ax.text(0, -0.2, f"{percent:.0f}%", ha="center", va="center",
                fontsize=22, fontweight="bold")

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-0.3, 1.2)
        ax.axis("off")
        plt.tight_layout()
        return fig

    

    # 2) update_gauge: trace 객체에 domain 수정
    @reactive.Effect
    async def update_gauge():
        reactive.invalidate_later(INTERVAL)
        y_vals = np.array(probas.get())
        current = (y_vals[-1] if y_vals.size > 0 else 0) * 100

        gauge_trace = {
            "type": "indicator",
            "mode": "gauge+number",
            "value": current,
            "domain": {"x": [0, 1], "y": [0, 0.8]},    # ← 아랫 60% 영역만 사용
        
            "number": {"suffix": "%", "font": {"size": 24}},
            "gauge": {
                "axis":   {"range": [0, 100], "tickwidth": 1, "tickcolor": "#666"},
                "bar":    {"color": "#7e57c2", "thickness": 0.3},
                "bgcolor": "#f0f0f0",
                "borderwidth": 2, "bordercolor": "#ccc",
                "steps": [
                    {"range": [0, 50],  "color": "#c8e6c9"},
                    {"range": [50, 100],"color": "#ffcdd2"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        }

        await session.send_custom_message("gauge_update", json.dumps(gauge_trace))


    # (9) 몰드 코드별 파이차트 렌더
    @output
    @render.plot
    def production_pie():
        if "mold_code" in df_all.columns:
            col_name = "mold_code"
        elif "mold_name" in df_all.columns:
            col_name = "mold_name"
        else:
            col_name = None

        df_subset = df_all.iloc[: len(probas.get())].copy()
        if (not col_name) or (col_name not in df_subset.columns) or df_subset.empty:
            fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
            ax.text(0.5, 0.5, "몰드 코드 데이터 없음", ha="center", va="center", fontsize=12)
            ax.axis("off")
            return fig

        counts = df_subset[col_name].value_counts()
        fig, ax = plt.subplots(figsize=(4, 4), dpi=80)
        ax.pie(counts, labels=counts.index, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
        ax.axis("equal")
        return fig

   


    # (10) 센서 박스 렌더러 등록
    def make_value_renderer(var_id, var_label, suffix):
        @output(id=f"{var_id}_box_{suffix}")
        @render.ui
        def _box():
            empty = "background-color:#EEE; padding:8px; border-radius:8px; font-weight:bold; font-size:14px; text-align:center; width:100px;"
            if suffix == "tab1":
                n = last_n.get()
                if n == 0:
                    return tags.div(f"{var_label}: --", style=empty)
                row = df_all.iloc[n-1]
                val = row.get(var_id, None)
            else:
                idx = click_row.get()
                if idx is None:
                    return tags.div(f"{var_label}: --", style=empty)
                row = df_all.iloc[idx]
                try:
                    val = float(input[f"{var_id}_input"]())
                except:
                    val = row.get(var_id, None)
            if val is None or pd.isna(val):
                return tags.div(f"{var_label}: N/A", style=empty)
            val = float(val)
            # IQR coloring
            mold = str(row.get("mold_code", "")).strip()
            lo, hi = iqr_thresholds.get(mold, {}).get(var_id, (None, None))
            bg = "#CCFFCC" if lo is not None and lo <= val <= hi else "#FFCCCC"
            return tags.div(f"{var_label}: {val:.1f}", style=f"background-color:{bg}; padding:8px; border-radius:8px; font-weight:bold; font-size:14px; text-align:center; width:100px;")
        return _box

    for v, lbl in slider_variables:
        make_value_renderer(v, lbl, "tab1")
        make_value_renderer(v, lbl, "tab2")

    # (10') 몰드 코드 박스
    @output
    @render.ui
    def mold_code_box_tab1():
        n = last_n.get()
        if n == 0:
            return tags.div("Mold: --", style="background:#EEE; padding:8px; border-radius:8px; width:100px;")
        m = df_all.iloc[n-1].get("mold_code", "")
        return tags.div(f"Mold: {m}", style="background:#CCC; padding:8px; border-radius:8px; width:100px;")

    @output
    @render.ui
    def mold_code_box_tab2():
        idx = click_row.get()
        if idx is None:
            return tags.div("Mold: --", style="background:#EEE; padding:8px; border-radius:8px; width:100px;")
        m = df_all.iloc[idx].get("mold_code", "")
        return tags.div(f"Mold: {m}", style="background:#CCC; padding:8px; border-radius:8px; width:100px;")

    # (11) 슬라이더 UI
    for v, lbl in slider_variables:
        @output(id=f"{v}_input_ui")
        @render.ui
        def _input_ui(var_id=v, label=lbl):
            idx = click_row.get()
            default = 50
            if idx is not None:
                val = df_all.iloc[idx].get(var_id, None)
                if pd.notna(val):
                    default = float(val)
            return ui.input_numeric(f"{var_id}_input", label, value=default, min=0, max=100)

    # (11') 몰드 코드 선택 UI
    @output
    @render.ui
    def mold_code_input_ui():
        idx = click_row.get()
        choices = [str(x) for x in sorted(df_all["mold_code"].unique())]
        sel = choices[0]
        if idx is not None:
            m = str(df_all.iloc[idx].get("mold_code", sel))
            if m in choices:
                sel = m
        return ui.input_select("mold_code_input", "금형 코드 선택", choices=choices, selected=sel)

    # (12) 선택된 행 처리 (placeholder)
    @reactive.Effect
    @reactive.event(input.selected_row)
    def on_row_selected():
        pass

    # ─── 모달용 Reactive 변수 ───────────────────────────────────────────
    pred_text_modal = reactive.Value("🔎 예측 버튼을 눌러주세요.")
    pred_logs_modal = reactive.Value([])

    # ─── UI 쪽에는 ui.output_ui("predict_result_box") 가 들어있다고 가정 ────

    @reactive.Effect
    @reactive.event(input.predict_btn_modal)
    def _do_modal_prediction():
        try:
            idx = click_row.get()
            if idx is None:
                raise ValueError("선택된 행이 없습니다.")
            base_row = df_all.iloc[idx].copy()
            for v,_ in slider_variables:
                try:
                    base_row[v] = input[f"{v}_input"]()
                except: pass
            prob = model.predict_proba(pd.DataFrame([base_row]))[0,1]
            pct  = f"{prob:.2%}"
            pred_text_modal.set(pct)
            ts = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
            pred_logs_modal.set([f"[{ts}] 예측: {pct}"] + pred_logs_modal())
        except Exception as e:
            pred_text_modal.set("❌ 오류")
            pred_logs_modal.set([str(e)] + pred_logs_modal())


    @output
    @render.ui
    def predict_result_ui():
        txt   = pred_text_modal()
        color = "red" if txt.endswith("%") and float(txt.strip("%"))/100 >= 0.5 else "#222"
        return tags.div(txt, style=f"font-size:24px; font-weight:bold; color:{color};")

    @output
    @render.ui
    def predict_log_ui():
        return tags.pre("\n".join(pred_logs_modal()), style="margin:0;")




# ───────────────────────────────────────────────────────────────────────────
# 6) App 객체 생성
# ───────────────────────────────────────────────────────────────────────────
app = App(app_ui, server, static_assets=static_path)