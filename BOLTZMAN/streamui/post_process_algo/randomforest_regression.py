import json
import pickle
import sys

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_curve
import streamlit as st
from mlxtend.plotting import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
from StorageManage.redis import redis_mgr
from multilanguage.multi_lang import item_caption
from streamlit import session_state as session
import pandas as pd

def get_rfr_model(session_id):
    key = f'rfr_report_{session_id}'
    try:
        model_key = redis_mgr.get(key)
        model_desc = pickle.loads(model_key)
        model_obj = model_desc['model']
        xt = model_desc['X_test']
        yt = model_desc['y_test']
        yp = model_desc['y_pred']
        return model_obj, xt, yt, yp
    except Exception as e:
        print(e)

def do_rfr_report(elk_logger, session_id, flexible=True, custom_df=None):
    key = f'rfr_{session_id}'
    elk_logger.debug({sys._getframe().f_code.co_name:f"do_rfr_report {key}"})
    model_status = redis_mgr.get(key)
    if model_status is not None:
        json_ms = json.loads(model_status)
        if json_ms['status'] == 'finish':
            key = f'rfr_report_{session_id}'
            try:
                model_key = redis_mgr.get(key)
                model_desc = pickle.loads(model_key)
                model_obj = model_desc['model']
                features = model_desc['features']
                features_importance = model_desc['features_importance']
                valid_r_square = model_desc['valid_r_square']
                valid_rmse = model_desc['valid_rmse']
                X_test = model_desc['X_test']
                yt = model_desc['y_test']
                yp = model_desc['y_pred']
                y_pred = model_desc['y_pred']

                if custom_df is not None:
                    st.write('사용자 데이터 분류 결과')
                    st.info(f"모델에 사용된 특성/독립변수")
                    st.info(features)
                    custom_xt = custom_df[features]
                    custome_predict_np = model_obj.predict(custom_xt)
                    predict_df = pd.DataFrame(custome_predict_np, columns=['Predict'])
                    result_df = pd.concat([custom_xt, predict_df], axis=1)
                    st.dataframe(result_df)
                st.write("Model: ", json_ms['name'])
                st.write("R²: ", valid_r_square)
                st.write("RMSE: ", valid_rmse)

                st.subheader("특성값 중요도")
                rfr_importances = model_obj.feature_importances_

                std = np.std([tree.feature_importances_ for tree in model_obj.estimators_], axis=0)
                forest_importances = pd.Series(rfr_importances, index=features)
                forest_importances_df = forest_importances.to_frame()
                forest_importances_df.columns = ["중요도"]
                st.dataframe(forest_importances_df)

                fig, ax = plt.subplots(figsize=(7, 4))
                forest_importances.plot.bar(yerr=std, ax=ax)
                ax.set_xticklabels(labels=features, rotation=90)
                ax.set_ylabel(f"{item_caption['mean_dec_imp'][session['LANG']]}")
                st.write(fig)

            except Exception as e:
                st.error(str(e))

