import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
import numpy as np
import joblib
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# 1. 데이터셋 준비
train = pd.read_csv("C:/Users/LS/Desktop/프로젝트_2/data/train.csv")
train = train[train['id'] != 19327]
train = train[
    (train['emergency_stop'].notna()) &
    (train['tryshot_signal'] != 'D') &
    (train['mold_code'].isin([8722, 8412, 8917]))
].copy()

drop_cols = [
    'id', 'line', 'name', 'mold_name', 'time', 'date', 'working',
    'emergency_stop', 'facility_operation_cycleTime', 'production_cycletime',
    'molten_volume', 'upper_mold_temp3', 'lower_mold_temp3', 'EMS_operation_time',
    'registration_time', 'passorfail', 'tryshot_signal', 'heating_furnace'
]
train_X = train.drop(drop_cols, axis=1)
train_X['mold_code'] = train_X['mold_code'].astype(str)
train_y = train['passorfail']

# 2. 명목형/수치형 변수 분리
num_columns = train_X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_columns = ['mold_code']

# 3. 학습/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(
    train_X, train_y,
    test_size=0.2,
    random_state=42,
    stratify=train_y
)

# 4. 전처리
preprocess = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num_columns),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_columns)
])

# 5. 파이프라인 및 학습
full_pipe = Pipeline([
    ("preprocess", preprocess),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': np.arange(100, 500, 100),
    'classifier__max_features': ['sqrt']
}

grid_search = GridSearchCV(
    estimator=full_pipe,
    param_grid=param_grid,
    cv=5,
    scoring='recall',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
y_pred = grid_search.predict(X_test)

joblib.dump(grid_search.best_estimator_, "C:/Users/LS/Desktop/프로젝트_2/best_model.pkl")

# 6. 평가
print(classification_report(y_test, y_pred, digits=4))
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred))
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# 7. SHAP 분석
best_pipe = grid_search.best_estimator_
preprocessor = best_pipe.named_steps["preprocess"]
model = best_pipe.named_steps["classifier"]

# SHAP 입력에 필요한 컬럼 이름 재구성
X_test_transformed = preprocessor.transform(X_test)

# OneHotEncoder 적용 후 컬럼 이름 추출
num_feature_names = num_columns
cat_feature_names = preprocessor.named_transformers_["cat"].get_feature_names_out(cat_columns)
all_feature_names = np.concatenate([num_feature_names, cat_feature_names])

X_test_df = pd.DataFrame(X_test_transformed.toarray() if hasattr(X_test_transformed, "toarray") else X_test_transformed,
                         columns=all_feature_names)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test_df)

shap.summary_plot(shap_values[:,:,1], X_test_df, feature_names=all_feature_names)

# 변수명 한글 매핑
variable_name_map = {
    "molten_temp": "용탕 온도",
    "low_section_speed": "하단 구간 속도",
    "high_section_speed": "상단 구간 속도",
    "cast_pressure": "주조 압력",
    "upper_mold_temp1": "상부 몰드 온도1",
    "upper_mold_temp2": "상부 몰드 온도2",
    "lower_mold_temp1": "하부 몰드 온도1",
    "lower_mold_temp2": "하부 몰드 온도2",
    "biscuit_thickness": "비스킷 두께",
    "sleeve_temperature": "슬리브 온도",
    "physical_strength": "물리적 강도",
    "Coolant_temperature": "냉각수 온도",
    "mold_code_8412": "몰드 코드 8412",
    "mold_code_8722": "몰드 코드 8722",
    "mold_code_8917": "몰드 코드 8917",
    "count": "일자별 생산 번호"
}

# 1. SHAP 중요도 계산 (평균 절대값)
shap_importance = np.abs(shap_values[:, :, 1]).mean(axis=0)

# 2. 변수명을 한글로 변환
korean_feature_names = [variable_name_map.get(feat, feat) for feat in all_feature_names]

# 3. DataFrame 생성 (한글 변수명 사용)
shap_df = pd.DataFrame({
    "변수명": korean_feature_names,
    "평균_절대_SHAP값": shap_importance
})

# 4. 중요도 순 정렬
shap_df_sorted = shap_df.sort_values(by="평균_절대_SHAP값", ascending=False)

# 5. Plotly로 인터랙티브 막대그래프 생성
import plotly.express as px

fig = px.bar(
    shap_df_sorted,
    x="평균_절대_SHAP값",
    y="변수명",
    orientation='h',
    title="SHAP 변수 중요도 (평균 절대값)",
    labels={"평균_절대_SHAP값": "SHAP 평균 절대값", "변수명": "변수"},
    height=740
)
fig.update_layout(yaxis=dict(autorange="reversed"))

fig.update_traces(
    hovertemplate="<b>%{y}</b><br>SHAP 평균 절대값: %{x:.3f}<extra></extra>"
)

fig.show()

# 6. 결과를 HTML 파일로 저장
fig.write_html("C:/Users/LS/Desktop/프로젝트_2/shap_feature_importance.html")
