import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

class MPGAnalyzer:
    def __init__(self, csv_path: str):
        """
        MPG 데이터 분석을 위한 클래스 초기화
        
        Args:
            csv_path: MPG CSV 파일 경로
        """
        self.csv_path = csv_path
        self.df = None
        self.df_with_missing = None
        
    def load_data(self) -> pd.DataFrame:
        """
        CSV 파일을 로드하고 파생 컬럼 생성
        
        Returns:
            로드된 데이터프레임
        """
        self.df = pd.read_csv(self.csv_path)
        
        # 파생 컬럼 생성: 평균 연비 (avg_mpg)
        self.df['avg_mpg'] = (self.df['cty'] + self.df['hwy']) / 2
        
        print(f"데이터 로드 완료: {len(self.df)}행 × {len(self.df.columns)}열")
        print(f"생성된 파생 컬럼: avg_mpg (cty와 hwy의 평균)")
        
        return self.df
    
    def generate_missing_values(self, missing_rate: float = 0.1) -> pd.DataFrame:
        """
        cty와 hwy 열에 인위적으로 결측치 생성
        
        Args:
            missing_rate: 결측치 비율 (기본값: 0.1)
            
        Returns:
            결측치가 포함된 데이터프레임
        """
        self.df_with_missing = self.df.copy()
        
        # cty와 hwy에 무작위로 결측치 생성
        n_missing = int(len(self.df_with_missing) * missing_rate)
        
        # cty 열에 결측치 생성
        cty_missing_idx = np.random.choice(self.df_with_missing.index, 
                                          size=n_missing, 
                                          replace=False)
        self.df_with_missing.loc[cty_missing_idx, 'cty'] = np.nan
        
        # hwy 열에 결측치 생성  
        hwy_missing_idx = np.random.choice(self.df_with_missing.index, 
                                          size=n_missing, 
                                          replace=False)
        self.df_with_missing.loc[hwy_missing_idx, 'hwy'] = np.nan
        
        print(f"결측치 생성 완료:")
        print(f"- cty 결측치: {self.df_with_missing['cty'].isnull().sum()}개")
        print(f"- hwy 결측치: {self.df_with_missing['hwy'].isnull().sum()}개")
        
        return self.df_with_missing
    
    def handle_missing_values(self, method: str = 'mean') -> Dict[str, pd.DataFrame]:
        """
        다양한 방법으로 결측치 처리
        
        Args:
            method: 처리 방법 ('drop', 'mean', 'median', 'manufacturer_mean')
            
        Returns:
            처리 방법별 데이터프레임 딕셔너리
        """
        if self.df_with_missing is None:
            raise ValueError("결측치가 생성된 데이터가 없습니다. generate_missing_values()를 먼저 실행하세요.")
        
        results = {}
        
        # 1. 결측치 제거
        df_dropped = self.df_with_missing.dropna(subset=['cty', 'hwy']).copy()
        df_dropped['avg_mpg'] = (df_dropped['cty'] + df_dropped['hwy']) / 2
        results['dropped'] = df_dropped
        print(f"결측치 제거 후: {len(df_dropped)}행 (제거된 행: {len(self.df_with_missing) - len(df_dropped)}개)")
        
        # 2. 평균값으로 대체
        df_mean = self.df_with_missing.copy()
        df_mean['cty'].fillna(df_mean['cty'].mean(), inplace=True)
        df_mean['hwy'].fillna(df_mean['hwy'].mean(), inplace=True)
        df_mean['avg_mpg'] = (df_mean['cty'] + df_mean['hwy']) / 2
        results['mean'] = df_mean
        print(f"평균값 대체 완료")
        
        # 3. 중앙값으로 대체
        df_median = self.df_with_missing.copy()
        df_median['cty'].fillna(df_median['cty'].median(), inplace=True)
        df_median['hwy'].fillna(df_median['hwy'].median(), inplace=True)
        df_median['avg_mpg'] = (df_median['cty'] + df_median['hwy']) / 2
        results['median'] = df_median
        print(f"중앙값 대체 완료")
        
        # 4. 제조사별 평균으로 대체
        df_manufacturer = self.df_with_missing.copy()
        
        # 제조사별 평균 계산
        manufacturer_means = df_manufacturer.groupby('manufacturer')[['cty', 'hwy']].mean()
        
        # 결측치를 제조사별 평균으로 대체
        for manufacturer in df_manufacturer['manufacturer'].unique():
            mask = df_manufacturer['manufacturer'] == manufacturer
            
            cty_mean = manufacturer_means.loc[manufacturer, 'cty']
            hwy_mean = manufacturer_means.loc[manufacturer, 'hwy']
            
            df_manufacturer.loc[mask & df_manufacturer['cty'].isnull(), 'cty'] = cty_mean
            df_manufacturer.loc[mask & df_manufacturer['hwy'].isnull(), 'hwy'] = hwy_mean
        
        df_manufacturer['avg_mpg'] = (df_manufacturer['cty'] + df_manufacturer['hwy']) / 2
        results['manufacturer_mean'] = df_manufacturer
        print(f"제조사별 평균 대체 완료")
        
        return results
    
    def analyze_by_manufacturer(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        제조사별 평균 도시 연비와 고속도로 연비 계산
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            제조사별 연비 통계
        """
        manufacturer_stats = df.groupby('manufacturer').agg({
            'cty': 'mean',
            'hwy': 'mean',
            'avg_mpg': 'mean'
        }).round(2)
        
        manufacturer_stats.columns = ['평균_도시연비', '평균_고속도로연비', '평균_연비']
        manufacturer_stats = manufacturer_stats.sort_values('평균_연비', ascending=False)
        
        print(f"제조사별 연비 분석 완료 ({len(manufacturer_stats)}개 제조사)")
        return manufacturer_stats
    
    def analyze_by_class(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        차량 카테고리별 배기량의 평균 계산
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            카테고리별 배기량 통계
        """
        class_stats = df.groupby('class').agg({
            'displ': ['mean', 'count']
        }).round(2)
        
        class_stats.columns = ['평균_배기량', '차량_수']
        class_stats = class_stats.sort_values('평균_배기량', ascending=False)
        
        print(f"차량 카테고리별 배기량 분석 완료 ({len(class_stats)}개 카테고리)")
        return class_stats
    
    def analyze_by_year(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        연도별 실린더 수의 변화 분석
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            연도별 실린더 통계
        """
        year_stats = df.groupby('year').agg({
            'cyl': ['mean', 'median', 'std', 'count']
        }).round(2)
        
        year_stats.columns = ['평균_실린더수', '중앙값_실린더수', '표준편차_실린더수', '차량_수']
        
        print(f"연도별 실린더 수 변화 분석 완료")
        return year_stats
    
    def create_manufacturer_mpg_chart(self, df: pd.DataFrame, save_path: str = None):
        """
        제조사별 평균 연비 막대 그래프 생성
        
        Args:
            df: 분석할 데이터프레임
            save_path: 저장 경로 (선택사항)
        """
        manufacturer_stats = self.analyze_by_manufacturer(df)
        
        plt.figure(figsize=(12, 8))
        
        # 막대 그래프 생성
        bars = plt.bar(range(len(manufacturer_stats)), 
                      manufacturer_stats['평균_연비'],
                      color='skyblue', 
                      alpha=0.7)
        
        # 그래프 꾸미기
        plt.xlabel('제조사', fontsize=12)
        plt.ylabel('평균 연비 (mpg)', fontsize=12)
        plt.title('제조사별 평균 연비 비교', fontsize=14, fontweight='bold')
        plt.xticks(range(len(manufacturer_stats)), 
                  manufacturer_stats.index, 
                  rotation=45, 
                  ha='right')
        
        # 값 표시
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=9)
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"제조사별 평균 연비 차트 저장: {save_path}")
        
        plt.show()
    
    def create_class_displacement_chart(self, df: pd.DataFrame, save_path: str = None):
        """
        차량 카테고리와 평균 배기량 선 그래프 생성
        
        Args:
            df: 분석할 데이터프레임
            save_path: 저장 경로 (선택사항)
        """
        class_stats = self.analyze_by_class(df)
        
        plt.figure(figsize=(12, 8))
        
        # 선 그래프 생성
        plt.plot(range(len(class_stats)), 
                class_stats['평균_배기량'], 
                marker='o', 
                linewidth=2, 
                markersize=8,
                color='red',
                alpha=0.7)
        
        # 그래프 꾸미기
        plt.xlabel('차량 카테고리', fontsize=12)
        plt.ylabel('평균 배기량 (L)', fontsize=12)
        plt.title('차량 카테고리별 평균 배기량', fontsize=14, fontweight='bold')
        plt.xticks(range(len(class_stats)), 
                  class_stats.index, 
                  rotation=45, 
                  ha='right')
        
        # 값 표시
        for i, value in enumerate(class_stats['평균_배기량']):
            plt.text(i, value + 0.1, 
                    f'{value:.1f}L',
                    ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"차량 카테고리별 평균 배기량 차트 저장: {save_path}")
        
        plt.show()
    
    def generate_analysis_report(self, df: pd.DataFrame) -> str:
        """
        데이터 분석 결과에 대한 종합 보고서 생성
        
        Args:
            df: 분석할 데이터프레임
            
        Returns:
            분석 보고서 문자열
        """
        report = []
        report.append("=" * 60)
        report.append("MPG 데이터셋 분석 보고서")
        report.append("=" * 60)
        
        # 기본 통계
        report.append(f"\\n1. 데이터 개요")
        report.append(f"   - 총 차량 수: {len(df)}대")
        report.append(f"   - 제조사 수: {df['manufacturer'].nunique()}개")
        report.append(f"   - 차량 카테고리 수: {df['class'].nunique()}개")
        report.append(f"   - 연도 범위: {df['year'].min()} - {df['year'].max()}")
        
        # 연비 통계
        report.append(f"\\n2. 연비 분석")
        report.append(f"   - 평균 도시 연비: {df['cty'].mean():.1f} mpg")
        report.append(f"   - 평균 고속도로 연비: {df['hwy'].mean():.1f} mpg")
        report.append(f"   - 평균 종합 연비: {df['avg_mpg'].mean():.1f} mpg")
        
        # 제조사별 TOP 3
        manufacturer_stats = self.analyze_by_manufacturer(df)
        report.append(f"\\n3. 제조사별 연비 순위 (TOP 3)")
        for i, (manufacturer, stats) in enumerate(manufacturer_stats.head(3).iterrows()):
            report.append(f"   {i+1}. {manufacturer}: {stats['평균_연비']:.1f} mpg")
        
        # 차량 카테고리별 배기량
        class_stats = self.analyze_by_class(df)
        report.append(f"\\n4. 차량 카테고리별 배기량 (TOP 3)")
        for i, (vehicle_class, stats) in enumerate(class_stats.head(3).iterrows()):
            report.append(f"   {i+1}. {vehicle_class}: {stats['평균_배기량']:.1f}L")
        
        # 연도별 실린더 변화
        year_stats = self.analyze_by_year(df)
        report.append(f"\\n5. 연도별 실린더 수 변화")
        for year, stats in year_stats.iterrows():
            report.append(f"   {year}년: 평균 {stats['평균_실린더수']:.1f}개 실린더")
        
        report.append("\\n" + "=" * 60)
        
        return "\\n".join(report)