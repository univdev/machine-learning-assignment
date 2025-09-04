import os
import sys
import pandas as pd
import numpy as np
from data_processing import MPGAnalyzer

def main():
    """메인 실행 함수"""
    print("MPG 데이터셋 분석을 시작합니다...")
    print("=" * 60)
    
    # CSV 파일 경로 설정
    csv_path = 'mpg.csv'
    if not os.path.exists(csv_path):
        print(f"오류: {csv_path} 파일을 찾을 수 없습니다.")
        return
    
    # MPGAnalyzer 초기화
    analyzer = MPGAnalyzer(csv_path)
    
    # 1단계: 데이터 로드 및 파생 컬럼 생성
    print("\\n1단계: 데이터 로드 및 파생 컬럼 생성")
    print("-" * 40)
    df_original = analyzer.load_data()
    print(f"원본 데이터 샘플:")
    print(df_original[['manufacturer', 'model', 'cty', 'hwy', 'avg_mpg']].head())
    
    # 2단계: 결측치 인위 생성
    print("\\n2단계: 결측치 인위 생성")
    print("-" * 40)
    np.random.seed(42)  # 재현 가능한 결과를 위한 시드 설정
    df_with_missing = analyzer.generate_missing_values(missing_rate=0.15)
    
    # 3단계: 다양한 방법으로 결측치 처리
    print("\\n3단계: 결측치 처리")
    print("-" * 40)
    processed_dfs = analyzer.handle_missing_values()
    
    # 처리 방법별 결과 비교
    print("\\n결측치 처리 방법별 결과 요약:")
    for method, df in processed_dfs.items():
        avg_mpg_mean = df['avg_mpg'].mean()
        print(f"  - {method}: 평균 연비 = {avg_mpg_mean:.2f} mpg, 데이터 수 = {len(df)}행")
    
    # 4단계: 그룹핑 및 통계 분석 (평균값 대체 데이터 사용)
    analysis_df = processed_dfs['mean']  # 평균값으로 대체한 데이터 사용
    
    print("\\n4단계: 그룹핑 및 통계 분석")
    print("-" * 40)
    
    # 4-1: 제조사별 평균 도시 연비와 고속도로 연비
    print("\\n4-1. 제조사별 연비 분석")
    manufacturer_stats = analyzer.analyze_by_manufacturer(analysis_df)
    print(manufacturer_stats)
    
    # 4-2: 차량 카테고리별 배기량 평균
    print("\\n4-2. 차량 카테고리별 배기량 분석")
    class_stats = analyzer.analyze_by_class(analysis_df)
    print(class_stats)
    
    # 4-3: 연도별 실린더 수 변화
    print("\\n4-3. 연도별 실린더 수 변화 분석")
    year_stats = analyzer.analyze_by_year(analysis_df)
    print(year_stats)
    
    # 5단계: 데이터 시각화
    print("\\n5단계: 데이터 시각화")
    print("-" * 40)
    
    try:
        # 5-1: 제조사별 평균 연비 막대 그래프
        print("제조사별 평균 연비 막대 그래프 생성 중...")
        analyzer.create_manufacturer_mpg_chart(
            analysis_df, 
            save_path='../manufacturer_mpg_chart.png'
        )
        
        # 5-2: 차량 카테고리별 평균 배기량 선 그래프
        print("차량 카테고리별 평균 배기량 선 그래프 생성 중...")
        analyzer.create_class_displacement_chart(
            analysis_df,
            save_path='../class_displacement_chart.png'
        )
        
    except Exception as e:
        print(f"시각화 생성 중 오류 발생: {e}")
        print("GUI 환경이 아니거나 matplotlib 설정에 문제가 있을 수 있습니다.")
    
    # 6단계: 종합 분석 보고서
    print("\\n6단계: 종합 분석 보고서")
    print("-" * 40)
    report = analyzer.generate_analysis_report(analysis_df)
    print(report)
    
    # 보고서 파일로 저장
    with open('../analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    print("\\n분석 보고서가 'analysis_report.txt' 파일로 저장되었습니다.")
    
    # 7단계: 결측치 처리 방법별 비교 분석
    print("\\n7단계: 결측치 처리 방법별 비교")
    print("-" * 40)
    
    comparison_results = []
    for method, df in processed_dfs.items():
        manufacturer_mean = analyzer.analyze_by_manufacturer(df)
        top_manufacturer = manufacturer_mean.index[0]
        top_mpg = manufacturer_mean.iloc[0]['평균_연비']
        
        comparison_results.append({
            '처리방법': method,
            '데이터수': len(df),
            '전체평균연비': df['avg_mpg'].mean(),
            '최고연비제조사': top_manufacturer,
            '최고평균연비': top_mpg
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    comparison_df = comparison_df.round(2)
    print(comparison_df.to_string(index=False))
    
    print("\\n" + "=" * 60)
    print("MPG 데이터셋 분석이 완료되었습니다!")
    print("생성된 파일:")
    print("  - analysis_report.txt: 종합 분석 보고서")
    if os.path.exists('../manufacturer_mpg_chart.png'):
        print("  - manufacturer_mpg_chart.png: 제조사별 연비 차트")
    if os.path.exists('../class_displacement_chart.png'):
        print("  - class_displacement_chart.png: 카테고리별 배기량 차트")
    print("=" * 60)

if __name__ == "__main__":
    main()