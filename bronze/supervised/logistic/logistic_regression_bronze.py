import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 데이터 불러오기 및 필요한 열 선택
df = pd.read_csv('bronze.csv')
chemical_elements = ["Cu", "Sn", "Pb", "Zn", "Au", "Ag", "As", "Sb"]
df_selected = df[chemical_elements + ["GROUP"]].copy()

X = df_selected[chemical_elements]
y = df_selected['GROUP']

# 결측값 처리 및 스케일링
X.fillna(X.mean(), inplace=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 로지스틱 회귀 모델 학습
model = LogisticRegression(solver='liblinear', max_iter=200, random_state=42)
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)

print("--- Logistic Regression Model Evaluation ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# 계수 출력
print("\nLogistic Regression Coefficients (Feature Weights):")
if len(model.classes_) > 2:
    coefficients_df = pd.DataFrame(model.coef_, columns=chemical_elements, index=model.classes_)
    print(coefficients_df)
else:
    coefficients_df = pd.DataFrame({'Feature': chemical_elements, 'Coefficient': model.coef_[0]})
    print(coefficients_df)

# 개별 예측 결과 시각화 준비
X_test_original = scaler.inverse_transform(X_test)
results_df = pd.DataFrame(X_test_original, columns=chemical_elements)
results_df['Predicted_GROUP'] = y_pred
results_df['True_GROUP'] = y_test.reset_index(drop=True)
results_df['Is_Correct'] = (results_df['Predicted_GROUP'] == results_df['True_GROUP'])

# 시각화 1: Cu vs Sn (True vs Predicted)
plt.figure(figsize=(12, 9))
sns.scatterplot(x='Cu', y='Sn', hue='True_GROUP', style='Predicted_GROUP',
                data=results_df, palette='tab10', s=150, alpha=0.8, markers=True)
plt.title('Actual vs. Predicted Groups (Cu vs. Sn Content)')
plt.xlabel('Cu Content (%)')
plt.ylabel('Sn Content (%)')
plt.grid(True)
plt.legend(title='Group (Color: Actual, Style: Predicted)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 시각화 2: Pb vs Zn (True vs Predicted)
plt.figure(figsize=(12, 9))
sns.scatterplot(x='Pb', y='Zn', hue='True_GROUP', style='Predicted_GROUP',
                data=results_df, palette='tab10', s=150, alpha=0.8, markers=True)
plt.title('Actual vs. Predicted Groups (Pb vs. Zn Content)')
plt.xlabel('Pb Content (%)')
plt.ylabel('Zn Content (%)')
plt.grid(True)
plt.legend(title='Group (Color: Actual, Style: Predicted)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# 시각화 3: Cu vs Sn with correctness
plt.figure(figsize=(12, 9))
sns.scatterplot(x='Cu', y='Sn', hue='True_GROUP', style='Is_Correct',
                data=results_df, palette='tab10', s=150, alpha=0.8)
plt.title('Actual Groups (Cu vs. Sn) with Correctness Indication')
plt.xlabel('Cu Content (%)')
plt.ylabel('Sn Content (%)')
plt.grid(True)
plt.legend(title='Actual Group (Marker: Correct?)', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

