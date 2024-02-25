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

def get_rfc_model(session_id):
    key = f'rfc_report_{session_id}'
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

def do_rfc_report(elk_logger, session_id, flexible=True, custom_df=None):
    key = f'rfc_{session_id}'
    elk_logger.debug({sys._getframe().f_code.co_name:f"do_rfc_report {key}"})
    model_status = redis_mgr.get(key)
    if model_status is not None:
        json_ms = json.loads(model_status)
        if json_ms['status'] == 'finish':
            key = f'rfc_report_{session_id}'
            try:
                elk_logger.debug({"status": "ok", sys._getframe().f_code.co_name: f"rfc_session id:{key}"})
                model_key = redis_mgr.get(key)
                model_desc = pickle.loads(model_key)
                elk_logger.debug({"status": "ok", sys._getframe().f_code.co_name: f"model_desc:{model_desc}"})

                yt = model_desc['y_test']
                yp = model_desc['y_pred']
                features = model_desc['features']
                accuracy = model_desc['accuracy']
                class_lbl = model_desc['class_lbl']
                model_obj = model_desc['model']
                if custom_df is not None:
                    st.write('사용자 데이터 분류 결과')
                    st.info(f"모델에 사용된 특성/독립변수")
                    st.info(features)
                    custom_xt = custom_df[features]
                    custome_predict_np = model_obj.predict(custom_xt)
                    predict_df = pd.DataFrame(custome_predict_np, columns=['Predict'])
                    result_df = pd.concat([custom_xt, predict_df], axis=1)
                    st.dataframe(result_df)

                st.write("Model:", json_ms['name'])
                st.write("Accuracy: ", accuracy.round(2))
                st.write("Precision: ", precision_score(yt, yp, labels=class_lbl, average='micro').round(2))
                st.write("Recall: ", recall_score(yt, yp, labels=class_lbl, average='micro').round(2))

                if flexible == True:
                    width = st.slider("plot width",min_value= 1.0, max_value=25.0,step=0.5, value=7.0, key = f'rpt_{session_id}_sld_szw')
                    height = st.slider("plot height", min_value=1.0, max_value=25.0, step=0.5, value=4.0, key = f'rpt_{session_id}_sld_szd')
                else:
                    width = 7.0
                    height = 4.0

                st.subheader("Receiver Operating Characteristic (ROC) Curve")
                if len(class_lbl) > 2:
                    st.info(f"클래스가 {len(class_lbl)}개 입니다. ROC 커브는 이진 분류에서 사용됩니다.")
                else:
                    fig, ax = plt.subplots(figsize=(width, height))
                    fprs, tprs, th = roc_curve(yt, yp)
                    ax.plot([0,1],[0,1], label='STR')
                    ax.plot(fprs, tprs, label = 'ROC')
                    st.pyplot(fig)

                st.subheader("Confusion Matrix")
                cm = confusion_matrix(yt, yp, labels=class_lbl)
                fig, ax = plot_confusion_matrix(figsize=(width, height),
                                                conf_mat=cm,
                                                colorbar=True,
                                                show_absolute=False,
                                                show_normed=True,
                                                class_names=class_lbl)
                st.pyplot(fig)

                st.subheader("특성값 중요도")
                rfc_importances = model_obj.feature_importances_

                std = np.std([tree.feature_importances_ for tree in model_obj.estimators_], axis=0)
                forest_importances = pd.Series(rfc_importances, index=features)
                st.dataframe(forest_importances)

                x_angle = st.slider(f"{item_caption['angle_label'][session['LANG']]}", min_value=0.0, max_value=90.0, step=0.5, value=90.0,
                                    key=f'rpt_rfc_{session_id}_x_angle')
                fig, ax = plt.subplots(figsize=(width, height))
                forest_importances.plot.bar(yerr=std, ax=ax)
                ax.set_xticklabels(labels=features, rotation=x_angle)
                ax.set_ylabel(f"{item_caption['mean_dec_imp'][session['LANG']]}")
                st.write(fig)

                # show_dtree = st.checkbox('Decision Tree 시각화를 수행할까요? <시간이 오래 걸릴 수 있음>',value=False, key=f'do_rfc_report_checkbox_{session_id}')
                # if show_dtree:
                #     rf = model_desc['model']
                #     trlen = len(rf.estimators_)
                #     sel_tree_num = st.selectbox('Decision 트리 추적 번호', options=[x for x in range(trlen)], index = 0)
                #     sel_rf = rf.estimators_[sel_tree_num]
                #
                #     st.write(f'{trlen} 중 {sel_tree_num} 번째 Decision Tree')
                #     dot_data = export_graphviz(sel_rf
                #                                 , out_file=None
                #                                 , filled=True
                #                                 , rounded=True
                #                                 , special_characters=True)
                #     st.graphviz_chart(dot_data)
            except Exception as e:
                st.error(str(e))
                elk_logger.error("An exception was thrown!" + str(e))

