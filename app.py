from flask import Flask, render_template, jsonify
import pandas as pd
import random

app = Flask(__name__)

# 加载数据并清理列名中的空格
df_bin = pd.read_csv('Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', skipinitialspace=True)
df_bin.columns = df_bin.columns.str.strip()  # 去除列名中多余的空格

# 定义用于展示的列
display_columns = ["Source IP", "Source Port", "Destination IP", "Destination Port", 
                   "Protocol", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Flow IAT Std", "Label"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_alert')
def get_alert():
    # 随机选取一行数据
    random_data = df_bin.sample(1)[display_columns].to_dict(orient='records')[0]
    return jsonify(random_data)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)