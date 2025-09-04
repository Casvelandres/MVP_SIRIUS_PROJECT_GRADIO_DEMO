import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import joblib, io, base64

TITLE = "SiriusAI • Demo gratuita para entrenar un modelo con tu CSV"
DISCLAIMER = "⚠️ Uso demostrativo. No subas datos sensibles. Los archivos no se almacenan de forma persistente."

def train(csv_file, target_col, task):
    if csv_file is None:
        return "Por favor sube un archivo CSV.", None, None
    try:
        df = pd.read_csv(csv_file.name)
    except Exception as e:
        return f"Error al leer CSV: {e}", None, None

    if target_col not in df.columns:
        return f"La columna objetivo '{target_col}' no existe en el CSV.", None, None

    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = selector(dtype_include="number")(X)
    cat_cols = selector(dtype_exclude="number")(X)

    pre = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(), StandardScaler(with_mean=False)), num_cols),
        ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"),
                              OneHotEncoder(handle_unknown="ignore")), cat_cols),
    ])

    if task == "Clasificación":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=200, random_state=42)
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=42)
        }

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = []
    best_score, best_pipe, best_name = -1e9, None, None

    for name, model in models.items():
        pipe = make_pipeline(pre, model)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        if task == "Clasificación":
            f1 = f1_score(y_test, y_pred, average="weighted")
            acc = accuracy_score(y_test, y_pred)
            score = f1
            metrics = {"model": name, "accuracy": round(acc,4), "f1_weighted": round(f1,4)}
        else:
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            score = -mae
            metrics = {"model": name, "MAE": round(mae,4), "R2": round(r2,4)}
        results.append(metrics)
        if score > best_score:
            best_score, best_pipe, best_name = score, pipe, name

    buf = io.BytesIO()
    joblib.dump(best_pipe, buf)
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode()
    link = f'<a href="data:file/pkl;base64,{b64}" download="pipeline_{best_name}.pkl">Descargar modelo entrenado ({best_name})</a>'

    return pd.DataFrame(results), link, f"Mejor modelo: {best_name}"

with gr.Blocks(title="SiriusAI demo") as demo:
    gr.Markdown(f"# {TITLE}\n{DISCLAIMER}")
    with gr.Row():
        csv = gr.File(label="Sube tu CSV (≤ 30MB)", file_types=[".csv"])
        target = gr.Textbox(label="Nombre de columna objetivo (target)")
        task = gr.Radio(["Clasificación", "Regresión"], value="Clasificación", label="Tipo de tarea")
    btn = gr.Button("Entrenar")
    out_table = gr.Dataframe(label="Métricas por modelo")
    out_link = gr.HTML()
    out_best = gr.Markdown()
    btn.click(train, inputs=[csv, target, task], outputs=[out_table, out_link, out_best])

if __name__ == "__main__":
    demo.launch()