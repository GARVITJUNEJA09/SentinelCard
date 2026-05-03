import warnings
warnings.filterwarnings('ignore')

import queue
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

try:
    import ttkbootstrap as tb
    BOOTSTRAP = True
except Exception:
    BOOTSTRAP = False

RANDOM_STATE = 42


class ModernFraudApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Credit Card Fraud Detection System')
        self.root.geometry('1280x780')
        self.root.minsize(1100, 700)

        self.gui_queue = queue.Queue()
        self.is_running = False
        self.plot_canvases = []

        self.file_path = tk.StringVar()
        self.status_var = tk.StringVar(value='Ready')
        self.selected_model = tk.StringVar(value='Both Models')
        self.sampling_method = tk.StringVar(value='SMOTE')
        self.test_size_var = tk.StringVar(value='0.20')
        self.rf_trees_var = tk.StringVar(value='100')
        self.xgb_trees_var = tk.StringVar(value='150')

        self.configure_styles()
        self.build_layout()
        self.root.after(150, self.process_gui_queue)

    def configure_styles(self):
        if not BOOTSTRAP:
            style = ttk.Style()
            try:
                style.theme_use('clam')
            except Exception:
                pass
            style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'))
            style.configure('Sub.TLabel', font=('Segoe UI', 10))
            style.configure('Metric.TLabel', font=('Segoe UI', 11, 'bold'))
        else:
            style = ttk.Style()
            style.configure('Title.TLabel', font=('Segoe UI', 20, 'bold'))
            style.configure('Sub.TLabel', font=('Segoe UI', 10))
            style.configure('Metric.TLabel', font=('Segoe UI', 11, 'bold'))

    def build_layout(self):
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill='both', expand=True)

        header = ttk.Frame(outer)
        header.pack(fill='x', pady=(0, 10))
        ttk.Label(header, text='Credit Card Fraud Detection System', style='Title.TLabel').pack(anchor='w')
        ttk.Label(
            header,
            text='Interactive ML dashboard with preprocessing, imbalance handling, model comparison, and visual analytics.',
            style='Sub.TLabel'
        ).pack(anchor='w', pady=(2, 0))

        body = ttk.Frame(outer)
        body.pack(fill='both', expand=True)

        self.left_panel = ttk.Frame(body, padding=12)
        self.left_panel.pack(side='left', fill='y', padx=(0, 10))

        self.right_panel = ttk.Frame(body)
        self.right_panel.pack(side='left', fill='both', expand=True)

        self.build_controls()
        self.build_results_area()

    def build_controls(self):
        file_card = ttk.LabelFrame(self.left_panel, text='Dataset', padding=10)
        file_card.pack(fill='x', pady=(0, 10))
        ttk.Label(file_card, text='Select creditcard.csv').pack(anchor='w')
        ttk.Entry(file_card, textvariable=self.file_path, width=42).pack(fill='x', pady=6)
        ttk.Button(file_card, text='Browse CSV', command=self.browse_file).pack(fill='x')

        config_card = ttk.LabelFrame(self.left_panel, text='Configuration', padding=10)
        config_card.pack(fill='x', pady=(0, 10))

        ttk.Label(config_card, text='Model').pack(anchor='w')
        ttk.Combobox(config_card, textvariable=self.selected_model, values=['Both Models', 'Random Forest', 'XGBoost'], state='readonly').pack(fill='x', pady=(2, 8))

        ttk.Label(config_card, text='Sampling').pack(anchor='w')
        ttk.Combobox(config_card, textvariable=self.sampling_method, values=['SMOTE', 'None'], state='readonly').pack(fill='x', pady=(2, 8))

        ttk.Label(config_card, text='Test size').pack(anchor='w')
        ttk.Combobox(config_card, textvariable=self.test_size_var, values=['0.20', '0.25', '0.30'], state='readonly').pack(fill='x', pady=(2, 8))

        ttk.Label(config_card, text='Random Forest trees').pack(anchor='w')
        ttk.Entry(config_card, textvariable=self.rf_trees_var).pack(fill='x', pady=(2, 8))

        ttk.Label(config_card, text='XGBoost trees').pack(anchor='w')
        ttk.Entry(config_card, textvariable=self.xgb_trees_var).pack(fill='x', pady=(2, 8))

        action_card = ttk.LabelFrame(self.left_panel, text='Actions', padding=10)
        action_card.pack(fill='x', pady=(0, 10))
        self.run_btn = ttk.Button(action_card, text='Run Analysis', command=self.start_analysis)
        self.run_btn.pack(fill='x', pady=(0, 6))
        ttk.Button(action_card, text='Clear Output', command=self.clear_output).pack(fill='x')

        status_card = ttk.LabelFrame(self.left_panel, text='Status', padding=10)
        status_card.pack(fill='x', pady=(0, 10))
        ttk.Label(status_card, textvariable=self.status_var).pack(anchor='w')
        self.progress = ttk.Progressbar(status_card, mode='indeterminate')
        self.progress.pack(fill='x', pady=8)

        tips_card = ttk.LabelFrame(self.left_panel, text='Tips', padding=10)
        tips_card.pack(fill='x')
        ttk.Label(
            tips_card,
            text='• Use the Kaggle creditcard.csv file\n• SMOTE usually improves recall\n• Lower tree counts run faster\n• Metrics focus on F1 and AUPRC',
            justify='left'
        ).pack(anchor='w')

    def build_results_area(self):
        metrics_frame = ttk.Frame(self.right_panel)
        metrics_frame.pack(fill='x', pady=(0, 10))

        self.metric_labels = {}
        for title in ['Dataset Rows', 'Fraud Ratio', 'Best Model', 'Best AUPRC']:
            card = ttk.LabelFrame(metrics_frame, text=title, padding=10)
            card.pack(side='left', fill='x', expand=True, padx=(0, 8))
            value = ttk.Label(card, text='—', style='Metric.TLabel')
            value.pack(anchor='center', pady=8)
            self.metric_labels[title] = value

        self.notebook = ttk.Notebook(self.right_panel)
        self.notebook.pack(fill='both', expand=True)

        self.log_tab = ttk.Frame(self.notebook)
        self.metrics_tab = ttk.Frame(self.notebook)
        self.cm_tab = ttk.Frame(self.notebook)
        self.pr_tab = ttk.Frame(self.notebook)
        self.fi_tab = ttk.Frame(self.notebook)

        self.notebook.add(self.log_tab, text='Execution Log')
        self.notebook.add(self.metrics_tab, text='Metrics')
        self.notebook.add(self.cm_tab, text='Confusion Matrix')
        self.notebook.add(self.pr_tab, text='PR Curve')
        self.notebook.add(self.fi_tab, text='Feature Importance')

        self.log_box = tk.Text(self.log_tab, wrap='word', font=('Consolas', 10))
        self.log_box.pack(fill='both', expand=True)

        self.metrics_tree = ttk.Treeview(self.metrics_tab, columns=('Model', 'F1-score', 'AUPRC'), show='headings', height=12)
        for col in ('Model', 'F1-score', 'AUPRC'):
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, anchor='center', width=180)
        self.metrics_tree.pack(fill='both', expand=True, padx=10, pady=10)

        self.cm_notebook = ttk.Notebook(self.cm_tab)
        self.cm_notebook.pack(fill='both', expand=True)
        self.pr_notebook = ttk.Notebook(self.pr_tab)
        self.pr_notebook.pack(fill='both', expand=True)
        self.fi_notebook = ttk.Notebook(self.fi_tab)
        self.fi_notebook.pack(fill='both', expand=True)

    def browse_file(self):
        path = filedialog.askopenfilename(title='Select creditcard.csv', filetypes=[('CSV files', '*.csv'), ('All files', '*.*')])
        if path:
            self.file_path.set(path)

    def clear_output(self):
        self.log_box.delete('1.0', tk.END)
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
        for title in self.metric_labels:
            self.metric_labels[title].config(text='—')
        self.clear_notebook_tabs(self.cm_notebook)
        self.clear_notebook_tabs(self.pr_notebook)
        self.clear_notebook_tabs(self.fi_notebook)
        self.plot_canvases = []
        self.status_var.set('Ready')

    def clear_notebook_tabs(self, notebook):
        for tab_id in notebook.tabs():
            notebook.forget(tab_id)

    def add_figure_tab(self, notebook, tab_title, fig):
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=tab_title)
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        self.plot_canvases.append((canvas, fig))

    def log(self, message=''):
        self.log_box.insert(tk.END, message + '\n')
        self.log_box.see(tk.END)

    def queue_log(self, message):
        self.gui_queue.put(('log', message))

    def queue_status(self, message):
        self.gui_queue.put(('status', message))

    def queue_metric_card(self, key, value):
        self.gui_queue.put(('metric_card', (key, value)))

    def queue_metric_row(self, row):
        self.gui_queue.put(('metric_row', row))

    def queue_plot(self, plot_type, model_name, fig):
        self.gui_queue.put(('plot', (plot_type, model_name, fig)))

    def queue_done(self):
        self.gui_queue.put(('done', None))

    def queue_error(self, message):
        self.gui_queue.put(('error', message))

    def process_gui_queue(self):
        try:
            while True:
                event, payload = self.gui_queue.get_nowait()
                if event == 'log':
                    self.log(payload)
                elif event == 'status':
                    self.status_var.set(payload)
                elif event == 'metric_card':
                    key, value = payload
                    self.metric_labels[key].config(text=value)
                elif event == 'metric_row':
                    self.metrics_tree.insert('', tk.END, values=payload)
                elif event == 'plot':
                    plot_type, model_name, fig = payload
                    if plot_type == 'cm':
                        self.add_figure_tab(self.cm_notebook, model_name, fig)
                    elif plot_type == 'pr':
                        self.add_figure_tab(self.pr_notebook, model_name, fig)
                    elif plot_type == 'fi':
                        self.add_figure_tab(self.fi_notebook, model_name, fig)
                elif event == 'done':
                    self.is_running = False
                    self.run_btn.config(state='normal')
                    self.progress.stop()
                    self.status_var.set('Completed')
                    messagebox.showinfo('Completed', 'Analysis finished successfully.')
                elif event == 'error':
                    self.is_running = False
                    self.run_btn.config(state='normal')
                    self.progress.stop()
                    self.status_var.set('Error')
                    messagebox.showerror('Error', payload)
                    self.log(f'Error: {payload}')
        except queue.Empty:
            pass
        finally:
            self.root.after(150, self.process_gui_queue)

    def start_analysis(self):
        if self.is_running:
            messagebox.showinfo('Busy', 'Analysis is already running.')
            return
        path = self.file_path.get().strip()
        if not path:
            messagebox.showwarning('Missing file', 'Please select the Kaggle creditcard.csv file first.')
            return

        self.clear_output()
        self.is_running = True
        self.run_btn.config(state='disabled')
        self.progress.start(10)
        self.status_var.set('Starting...')

        worker = threading.Thread(target=self.run_worker, args=(path,), daemon=True)
        worker.start()

    def prepare_data(self, df, test_size):
        df = df.copy()
        scaler = RobustScaler()
        df[['Time', 'Amount']] = scaler.fit_transform(df[['Time', 'Amount']])
        X = df.drop(columns='Class')
        y = df['Class']
        return train_test_split(X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE)

    def build_pipeline(self, model):
        if self.sampling_method.get() == 'SMOTE':
            return ImbPipeline([('smote', SMOTE(random_state=RANDOM_STATE)), ('model', model)])
        return ImbPipeline([('model', model)])

    def build_models(self):
        rf_trees = int(self.rf_trees_var.get())
        xgb_trees = int(self.xgb_trees_var.get())

        models = {
            'Random Forest': self.build_pipeline(RandomForestClassifier(
                n_estimators=rf_trees,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight='balanced_subsample'
            )),
            'XGBoost': self.build_pipeline(XGBClassifier(
                n_estimators=xgb_trees,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                eval_metric='logloss',
                random_state=RANDOM_STATE,
                n_jobs=-1,
                tree_method='hist'
            ))
        }

        choice = self.selected_model.get()
        if choice == 'Both Models':
            return models
        return {choice: models[choice]}

    def build_confusion_figure(self, y_true, y_pred, model_name):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        im = ax.imshow(cm, cmap='Blues')
        fig.colorbar(im, ax=ax)
        ax.set_title(f'{model_name} - Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Fraud', 'Fraud'])
        ax.set_yticklabels(['Non-Fraud', 'Fraud'])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        fig.tight_layout()
        return fig

    def build_pr_figure(self, y_true, y_proba, model_name, auprc):
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(recall, precision, linewidth=2, color='purple', label=f'AUPRC = {auprc:.4f}')
        ax.set_title(f'{model_name} - Precision Recall Curve')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        return fig

    def build_feature_importance_figure(self, model, feature_names, model_name, top_n=15):
        estimator = model.named_steps['model']
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        features = np.array(feature_names)[indices]
        values = importances[indices]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(features, values, color='teal')
        ax.set_title(f'{model_name} - Top {top_n} Features')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        fig.tight_layout()
        return fig

    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test, feature_names):
        self.queue_status(f'Training {name}...')
        self.queue_log('\n' + '=' * 90)
        self.queue_log(f'Training {name}...')

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        f1 = f1_score(y_test, y_pred)
        auprc = average_precision_score(y_test, y_proba)
        report = classification_report(y_test, y_pred, digits=4)

        self.queue_log(f'{name} Results')
        self.queue_log('-' * 90)
        self.queue_log(f'F1-score: {f1:.4f}')
        self.queue_log(f'AUPRC:    {auprc:.4f}')
        self.queue_log('Classification Report:')
        self.queue_log(report)

        self.queue_metric_row((name, f'{f1:.4f}', f'{auprc:.4f}'))
        self.queue_plot('cm', name, self.build_confusion_figure(y_test, y_pred, name))
        self.queue_plot('pr', name, self.build_pr_figure(y_test, y_proba, name, auprc))
        self.queue_plot('fi', name, self.build_feature_importance_figure(model, feature_names, name))

        return {'Model': name, 'F1-score': f1, 'AUPRC': auprc}

    def run_worker(self, path):
        try:
            self.queue_status('Loading dataset...')
            self.queue_log('Loading dataset...')
            df = pd.read_csv(path)

            required = {'Time', 'Amount', 'Class'}
            if not required.issubset(df.columns):
                raise ValueError("CSV must contain 'Time', 'Amount', and 'Class' columns.")

            fraud_ratio = df['Class'].mean()
            self.queue_metric_card('Dataset Rows', f'{len(df):,}')
            self.queue_metric_card('Fraud Ratio', f'{fraud_ratio:.4%}')
            self.queue_log(f'Dataset shape: {df.shape}')
            self.queue_log('Class distribution (proportion):')
            self.queue_log(df['Class'].value_counts(normalize=True).to_string())

            test_size = float(self.test_size_var.get())
            self.queue_status('Preparing data...')
            X_train, X_test, y_train, y_test = self.prepare_data(df, test_size)
            feature_names = X_train.columns.tolist()

            models = self.build_models()
            results = []
            for name, model in models.items():
                results.append(self.evaluate_model(name, model, X_train, X_test, y_train, y_test, feature_names))

            results_df = pd.DataFrame(results).sort_values('AUPRC', ascending=False)
            best = results_df.iloc[0]
            self.queue_metric_card('Best Model', best['Model'])
            self.queue_metric_card('Best AUPRC', f"{best['AUPRC']:.4f}")
            self.queue_log('\nFinal ranking by AUPRC:')
            self.queue_log(results_df.to_string(index=False))
            self.queue_done()

        except Exception as e:
            self.queue_error(str(e))


if __name__ == '__main__':
    if BOOTSTRAP:
        root = tb.Window(themename='cosmo')
    else:
        root = tk.Tk()
    app = ModernFraudApp(root)
    root.mainloop()
