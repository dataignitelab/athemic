import json
import pickle
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from StorageManage.redis import redis_mgr
from multilanguage.multi_lang import item_caption
from streamlit import session_state as session
import logging
import logstash

def do_statistics_report(elk_logger, session_id, flexible=True):
    status_key = f'statistics_{session_id}'
    elk_logger.debug(f"do_statistics_report {status_key}")
    model_status = redis_mgr.get(status_key)
    if model_status is None: pass
    else:
        json_ms = json.loads(model_status)
        if json_ms['status'] == 'finish':
            key = f'statistics_report_{session_id}'
            try:
                model_key = redis_mgr.get(key)
                model_desc = pickle.loads(model_key)
                report_pkg = model_desc['report']

                for idx, report_item in enumerate(report_pkg):
                    describe = report_item['describe']
                    feature = report_item['feature']
                    showtype = report_item['showtype'][0]
                    report_value = report_item['showtype'][1]

                    if showtype == 'df':
                        if feature == None: st.subheader(describe)
                        st.dataframe(report_value)
                    elif showtype == 'hist':
                        if len(report_value):
                            if flexible == True:
                                width = st.slider("Plot 너비", min_value=1.0, max_value=25.0, step=0.5, value=7.0,
                                                  key=f'rpt__{idx}_{session_id}_sld_szw')
                                height = st.slider("Plot 높이", min_value=1.0, max_value=25.0, step=0.5, value=4.0,
                                                   key=f'rpt__{idx}_{session_id}_sld_szd')
                            else:
                                width = 7.0
                                height = 4.0
                            st.subheader(f"{feature} {item_caption['histogram_chart'][session['LANG']]}")
                            fig, ax = plt.subplots(figsize=(width, height))
                            report_value = pd.read_json(report_value, orient='index')
                            ax.hist(report_value, bins=20, label=feature)
                            ax.legend()
                            st.pyplot(fig)
                    elif showtype == 'pie':
                        if len(report_value) > 0:
                            if flexible == True:
                                width = st.slider("Plot 너비", min_value=1.0, max_value=25.0, step=0.5, value=7.0,
                                                  key=f'rpt__{idx}_{session_id}_sld_szw')
                                height = st.slider("Plot 높이", min_value=1.0, max_value=25.0, step=0.5, value=4.0,
                                                   key=f'rpt__{idx}_{session_id}_sld_szd')
                            else:
                                width = 7.0
                                height = 4.0

                            st.subheader(f"{feature} {item_caption['pie_chart'][session['LANG']]}")
                            item_list = [key for key, value in report_value.items()]
                            value_list = [value for key, value in report_value.items()]
                            fig, ax = plt.subplots(figsize=(width, height))
                            ax.pie(value_list, labels=item_list)
                            st.pyplot(fig)

                    elif showtype == 'bar':
                        if len(report_value) > 0:
                            if flexible == True:
                                width = st.slider("Plot 너비", min_value=1.0, max_value=25.0, step=0.5, value=7.0,
                                                  key=f'rpt__{idx}_{session_id}_sld_szw')
                                height = st.slider("Plot 높이", min_value=1.0, max_value=25.0, step=0.5, value=4.0,
                                                   key=f'rpt__{idx}_{session_id}_sld_szd')
                            else:
                                width = 7.0
                                height = 4.0
                            st.subheader(f"{feature} {item_caption['bar_chart'][session['LANG']]} ")
                            item_list = [ key for key, value in report_value.items()]
                            value_list = [ value for key, value in report_value.items()]
                            x_angle = st.slider("x angle ", min_value=0.0, max_value=90.0, step=0.5, value=0.0,key=f'statistics_bar_{idx}_{session_id}_slider')
                            fig, ax = plt.subplots(figsize=(width, height))
                            ax.bar(item_list, value_list)
                            ax.set_xticklabels(item_list, rotation=x_angle)
                            st.pyplot(fig)
                return 0
            except Exception as e:
                elk_logger.error(str(e) + " Error")
                st.error(str(e)+" Error")
