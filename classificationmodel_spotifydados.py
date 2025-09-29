import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

# função utilitária para mostrar primeiros registros limpos do dataset
def show_head(df, name="DataFrame", n=5):
    try:
        from caas_jupyter_tools import display_dataframe_to_user
        display_dataframe_to_user(name, df.head(n))
    except Exception:
        print(df.head(n))

RANDOM_SEED = 42
DATA_PATH = "C:/Users/heito/PycharmProjects/PortifolioPython/spotify_churn_dataset.csv"  # caminho fornecido pelo usuário/upload
OUTPUT_MODEL_PATH = "C:/Users/heito/PycharmProjects/PortifolioPython/best_model.pkl"

print("1) Carregando dataset:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dimensão:", df.shape)
print("Colunas:", df.columns.tolist())

# identificar coluna alvo automaticamente
possible_targets = ['churn', 'Churn', 'is_churn', 'isChurn', 'label', 'target', 'Exited', 'cancelled']
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    # fallback: tentar coluna do tipo binário ou a última coluna
    # preferir colunas com apenas 0/1 ou True/False
    for col in df.columns[::-1]:
        unique_vals = df[col].dropna().unique()
        if set(unique_vals).issubset({0,1}) or set([str(x).lower() for x in unique_vals]).issubset({'true','false','0','1'}):
            target_col = col
            break
if target_col is None:
    # última coluna como última alternativa
    target_col = df.columns[-1]

print("Coluna alvo escolhida:", target_col)

# Mostrar resumo inicial
print("\nResumo inicial:")
print(df[target_col].value_counts(dropna=False))
show_head(df, "Dataset - amostra", n=6)

# Separar X e y
y = df[target_col].copy()
X = df.drop(columns=[target_col])

# Tratamento simples do alvo para valores textuais
if y.dtype == object or y.dtype.name == 'category':
    y = y.astype(str).str.strip().str.lower()
    # mapear valores comuns para 0/1
    mapping = {}
    if set(y.unique()).issubset({'yes','no','y','n','true','false','1','0','sim','nao','não'}):
        mapping = {v: 1 if v in ['yes','y','true','1','sim'] else 0 for v in y.unique()}
        # lidar com portuguese 'não' com acento
        mapping = {k: (1 if str(k).lower() in ['yes','y','true','1','sim'] else 0) for k in y.unique()}
    else:
        # caso generalizado: fatorizar e usar ultima classe como '1'
        _, y = pd.factorize(y)
else:
    # garantir binário inteiro 0/1 quando possível
    if set(y.dropna().unique()).issubset({0,1}):
        y = y.astype(int)
    else:
        # tentar binarizar com threshold (não ideal, apenas fallback)
        if y.nunique() <= 10:
            # mapear categorias
            y = pd.factorize(y)[0]
        else:
            # contínuo — transformar em binário pela mediana (fallback)
            med = y.median()
            y = (y > med).astype(int)

print("\nAlvo após processamento — distribuição:")
print(pd.Series(y).value_counts())

# Identificar tipos de colunas
num_cols = X.select_dtypes(include=['number']).columns.tolist()
cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
# tentar detecção adicional: colunas com poucas categorias tratadas como categóricas
for col in X.columns:
    if col not in num_cols and col not in cat_cols:
        # se poucas categorias únicas -> categórica
        if X[col].nunique(dropna=True) <= 15:
            cat_cols.append(col)
        else:
            num_cols.append(col)

# Remover colunas com alta cardinalidade textuais que não serão úteis (ex.: ids, urls) — heurística
drop_cols = []
for col in cat_cols:
    if X[col].nunique() > 200 and X[col].dtype == object:
        drop_cols.append(col)
if drop_cols:
    print("\nColunas com alta cardinalidade detectadas e removidas (heurística):", drop_cols)
    X = X.drop(columns=drop_cols)
    cat_cols = [c for c in cat_cols if c not in drop_cols]

print("\nColunas numéricas:", num_cols)
print("Colunas categóricas:", cat_cols)

# Pipeline de pré-processamento
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
], remainder='drop')  # descartamos colunas não listadas

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y if len(np.unique(y))>1 else None
)

print("\nTamanhos: X_train={}, X_test={}".format(X_train.shape, X_test.shape))

# Modelos a testar
models = {
    'logreg': LogisticRegression(max_iter=500, random_state=RANDOM_SEED),
    'rf': RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1),
    'gb': GradientBoostingClassifier(n_estimators=200, random_state=RANDOM_SEED)
}

# Treinar cada modelo com pipeline e avaliar
results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[('pre', preprocessor), ('clf', model)])
    print(f"\nTreinando: {name} ...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # probas para AUC (se disponível)
    try:
        y_proba = pipe.predict_proba(X_test)[:,1]
    except Exception:
        y_proba = None
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None and len(np.unique(y_test))>1 else None
    results[name] = {
        'pipeline': pipe,
        'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc,
        'y_pred': y_pred, 'y_proba': y_proba
    }
    print(f"Resultados {name}: ACC={acc:.4f} PREC={prec:.4f} REC={rec:.4f} F1={f1:.4f} AUC={auc if auc is None else format(auc, '.4f')}")


# Escolher melhor modelo por F1 (alternativa: AUC ou ACC)
best_name = max(results.keys(), key=lambda k: (results[k]['f1'], results[k]['accuracy']))
best = results[best_name]
print("\nMelhor modelo segundo F1: ", best_name)
print("Métricas detalhadas:\n", {k: v for k,v in results[best_name].items() if k in ['accuracy','precision','recall','f1','auc']})

# Relatório classification report
print("\nClassification report (melhor modelo):\n")
print(classification_report(y_test, best['y_pred'], zero_division=0))

# Matriz de confusão
cm = confusion_matrix(y_test, best['y_pred'])
print("Matriz de Confusão:\n", cm)

# Plot da matriz de confusão
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, interpolation='nearest')
ax.set_title("Matriz de Confusão - " + best_name)
ax.set_xlabel("Predito")
ax.set_ylabel("Real")
ax.xaxis.set_ticklabels([''] + list(np.unique(best['y_pred'])))
ax.yaxis.set_ticklabels([''] + list(np.unique(y_test)))
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center")
plt.tight_layout()
plt.show()

# Se houver probabilidades, plot ROC curve
if best['y_proba'] is not None and len(np.unique(y_test))>1:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, best['y_proba'])
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr)
    ax.plot([0,1],[0,1], linestyle='--')
    ax.set_title(f"ROC curve (AUC = {roc_auc:.4f}) - {best_name}")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    plt.show()

# Importância de features (quando disponível)
try:
    # extrair nomes de features após preprocessamento
    pre = best['pipeline'].named_steps['pre']
    # construir nomes das features transformadas
    feature_names = []
    if len(num_cols) > 0:
        feature_names += num_cols
    if len(cat_cols) > 0:
        ohe = pre.named_transformers_['cat'].named_steps['onehot']
        ohe_cols = list(ohe.get_feature_names_out(cat_cols))
        feature_names += ohe_cols
    # obter importâncias se o classificador suportar
    clf = best['pipeline'].named_steps['clf']
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        fi = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(30)
        print("\nTop 30 features por importância:\n", fi)
        # plot
        fig, ax = plt.subplots(figsize=(8,6))
        fi.plot(kind='bar', ax=ax)
        ax.set_title("Feature importances - " + best_name)
        plt.tight_layout()
        plt.show()
    elif hasattr(clf, "coef_"):
        coefs = clf.coef_
        if coefs.ndim == 1:
            coefs = coefs
            fi = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False).head(30)
            print("\nTop 30 features pelo coeficiente absoluto:\n", fi)
        else:
            print("Coeficientes multiclasse; omitido resumo extensivo.")
except Exception as e:
    print("Não foi possível computar importâncias de features automaticamente:", str(e))

# Salvar o melhor pipeline em disco
try:
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        pickle.dump(best['pipeline'], f)
    print("\nModelo salvo em:", OUTPUT_MODEL_PATH)
except Exception as e:
    print("Erro ao salvar o modelo:", e)

# Resumo dos resultados para todos os modelos
summary = pd.DataFrame([
    {'model': name,
     'accuracy': results[name]['accuracy'],
     'precision': results[name]['precision'],
     'recall': results[name]['recall'],
     'f1': results[name]['f1'],
     'auc': results[name]['auc']}
    for name in results
]).sort_values(by='f1', ascending=False).reset_index(drop=True)

print("\nResumo comparativo de modelos:")
show_head(summary, "Resumo de Modelos", n=10)

# Se desejar exportar predição sobre todo o dataset:
try:
    full_pred = best['pipeline'].predict_proba(X)[:,1] if hasattr(best['pipeline'], "predict_proba") else best['pipeline'].predict(X)
    out_df = df.copy()
    out_df['_predicted_proba'] = full_pred
    out_csv = "/mnt/data/predictions_full_dataset.csv"
    out_df.to_csv(out_csv, index=False)
    print("Predições completas salvas em:", out_csv)
except Exception as e:
    print("Não foi possível gerar predições para o dataset completo:", e)

print("\nFim do processamento. Você pode abrir /mnt/data/best_model.pkl e /mnt/data/predictions_full_dataset.csv na sua máquina local se fizer o download dos arquivos gerados.")
