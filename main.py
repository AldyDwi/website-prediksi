from flask import Flask, render_template, request, redirect, url_for
import pickle
from sklearn import preprocessing
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score

app = Flask(__name__)

model = pickle.load(open("model2.pkl", "rb"))

# Set folder untuk upload file
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route untuk halaman utama
@app.route('/')
def home():
    return render_template('index.html')

# Route untuk halaman Model 
@app.route('/model')
def model_page():
    # Cek apakah ada file CSV yang diunggah
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if csv_files:

        # Menghitung classification report dari telkomsel data
        df = pd.read_csv('uploads/0_cth.csv')
        # Menghapus kolom 'ID Pelanggan'
        df.drop(columns=['ID Pelanggan'], inplace=True)
        # mengganti isi kolom Promosi Terakhir dan Churn menjadi 1 dan 0
        Promosi = df.replace(to_replace={'Promosi Terakhir': {'YA': 1, 'TIDAK': 0}})
        Churn = Promosi.replace(to_replace={'Churn': {'YA': 1, 'TIDAK': 0}})
        data2 = Churn

        # menghitung outlier pada tagihan rata-rata
        Q1, Q3 = np.percentile(data2["Tagihan Rata-rata"], [25, 75])
        selisih = Q3 - Q1
        # menghitung batas bawah dan atas
        bw4 = Q1 - 1.5 * selisih
        ba4 = Q3 + 1.5 * selisih

        # Menghitung max dari nilai Tagihan Rata-rata yang bukan outlier
        rerata = data2[(data2["Tagihan Rata-rata"] >= bw4) & (data2["Tagihan Rata-rata"] <= ba4)]["Tagihan Rata-rata"].max()
        # Mengganti outlier dengan max
        data2.loc[(data2["Tagihan Rata-rata"] < bw4) | (data2["Tagihan Rata-rata"] > ba4), "Tagihan Rata-rata"] = rerata

        X_before = data2.drop(['Churn'], axis=1)
        y = data2['Churn']
        # Normalisasi fitur
        X = preprocessing.normalize(X_before)
        # buat fungsi K fold
        kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        # Membuat model RF
        rf = RandomForestClassifier(random_state=45)
        scores2 = []
        node_weights = []

        for train_indices, test_indices in kf.split(X, y):
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]

            # Fit model
            rf.fit(X_train, y_train)

            # Prediksi
            y_pred = rf.predict(X_test)

            # Simpan akurasi dari fold saat ini
            fold_score = accuracy_score(y_test, y_pred)
            scores2.append(fold_score)

            # Simpan bobot node untuk visualisasi
            node_weights.append(rf.feature_importances_)

        # Mean accuracy
        mean_accuracy2 = np.mean(scores2)
        avg_node_weight = np.mean(node_weights, axis=0)

        # Koordinat untuk visualisasi
        tree_num = len(avg_node_weight)
        tree_x = np.arange(tree_num)  # Indeks fitur
        # Gabungkan data ke dalam list of dicts
        data_for_chart = [{'x': x, 'y': y} for x, y in zip(tree_x, avg_node_weight)]

        return render_template('model.html', data_for_chart=data_for_chart)
    else:
        return render_template('model.html', message="Belum ada data.", df=None)

# Route untuk Prediction
@app.route('/prediction')
def prediction_page():
    return render_template('prediction.html')

# Route untuk newData
@app.route('/new-data')
def new_data_page():
    # Cek apakah ada file CSV yang diunggah
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if csv_files:
        # Inisialisasi DataFrame kosong
        combined_df = pd.DataFrame()
        
        # Baca setiap file CSV dan gabungkan ke dalam satu DataFrame
        for file in csv_files:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file))
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        combined_df['Churn'] = combined_df['Churn'].replace({1: 'YA', 0: 'TIDAK'})

        return render_template('new_data.html', df=combined_df)
    else:
        return render_template('new_data.html', message="Belum ada data.", df=None)



# Endpoint untuk upload file
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.csv'):

        # Beri nama unik pada file yang diunggah
        filename = secure_filename(file.filename)
        unique_filename = f"{len(os.listdir(UPLOAD_FOLDER)) + 1}_{filename.rsplit('.', 1)[0]}.csv"
        
        # Simpan file ke folder uploads
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)
        # Membaca file CSV menggunakan Pandas
        df = pd.read_csv(filepath)
        # Tampilkan isi file ke halaman web

        # Ganti nilai pada kolom "Promosi Terakhir"
        df['Promosi Terakhir'] = df['Promosi Terakhir'].replace({'YA': 1, 'TIDAK': 0})
        
        # Menyiapkan fitur untuk prediksi
        features = df[['Durasi Langganan (bulan)', 'Jumlah Panggilan', 'Penggunaan Data (GB)', 'Tagihan Rata-rata', 'Promosi Terakhir']].values

        # Lakukan normalisasi pada fitur
        features_normalized = preprocessing.normalize(features)
        
        # Lakukan prediksi dan tambahkan hasil prediksi sebagai kolom baru
        df['Churn'] = model.predict(features_normalized)
        
        # Ganti nilai kolom prediksi kembali ke bentuk aslinya
        df['Promosi Terakhir'] = df['Promosi Terakhir'].replace({1: 'YA', 0: 'TIDAK'})
        df['Churn'] = df['Churn'].replace({1: 'YA', 0: 'TIDAK'})
        
        # Simpan kembali file CSV yang telah diperbarui dengan kolom Churn
        updated_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_filename}")
        df.to_csv(updated_filepath, index=False)

        return render_template('new_data.html', df=df,)
    return redirect(request.url)


# Kode untuk memprediksi pelanggan churn pada halaman Prediction
@app.route('/prediction', methods=['POST'])
def predict():
    try:
        # Ambil semua input dari form dengan validasi
        durasi = request.form.get('durasi')
        jml_panggilan = request.form.get('jml_panggilan')
        guna_data = request.form.get('guna_data')
        mean_tagihan = request.form.get('mean_tagihan')
        promosi = request.form.get('promosi')

        # Pastikan semua field diisi dan bisa dikonversi ke integer
        if not all([durasi, guna_data, jml_panggilan, mean_tagihan, promosi]):
            return render_template("prediction.html", error="Error: Semua field harus diisi.", prediction_text=None)
        
        # Konversi input ke integer
        durasi = int(durasi)
        jml_panggilan = int(jml_panggilan)
        guna_data = int(guna_data)
        mean_tagihan = int(mean_tagihan)
        promosi = int(promosi)
        
        # Masukkan input ke dalam array
        fitur = np.array([[durasi, jml_panggilan, guna_data, mean_tagihan, promosi]])

        new_data = preprocessing.normalize(fitur)
        
        # Lakukan prediksi
        prediction = model.predict(new_data)

        # Prediksi probabilitas churn
        prediction_proba = model.predict_proba(new_data)

        # Tampilkan hasil prediksi di tab "Prediction"
        return render_template("prediction.html", prediction_text=f"{'Churn' if prediction == 1 else 'Tidak Churn' if prediction == 0 else None}", churn=f"{math.floor(prediction_proba[0][1] * 100)}", tidak=f"{math.floor(prediction_proba[0][0] * 100)}" )
    
    except ValueError:
        return render_template("prediction.html", error="Error: Input harus berupa angka.", prediction_text=None)
    except Exception as e:
        return render_template("prediction.html", error=f"Error: {str(e)}", prediction_text=None)

# Route untuk newData
@app.route('/comparison')
def comparison_page():
    # Cek apakah ada file CSV yang diunggah
    csv_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.csv')]
    if csv_files:

        # data lama
        df = pd.read_csv('uploads/0_cth.csv')
        #kolom 'churn' yang berisi 1 untuk churn dan 0 untuk tidak churn
        df['Churn'] = df['Churn'].replace({'YA': 1, 'TIDAK': 0})

        iya = df['Churn'].sum()  # Jumlah churn
        tidak = len(df) - iya  # Jumlah tidak churn

        # Hitung persentase
        total = len(df)
        yaLama = round((iya / total) * 100, 2)  # Persentase churn
        tidakLama = round((tidak / total) * 100, 2)  # Persentase tidak churn
        df['Churn'] = df['Churn'].replace({1: 'YA', 0: 'TIDAK'})



        # gabungan setelah upload data baru
        # Inisialisasi DataFrame
        combined_df = pd.DataFrame()
        
        # Baca setiap file CSV dan gabungkan ke dalam satu DataFrame
        for file in csv_files:
            df = pd.read_csv(os.path.join(UPLOAD_FOLDER, file))
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        
        #kolom 'churn' yang berisi 1 untuk churn dan 0 untuk tidak churn
        combined_df['Churn'] = combined_df['Churn'].replace({'YA': 1, 'TIDAK': 0})

        churn_count = combined_df['Churn'].sum()  # Jumlah churn
        not_churn_count = len(combined_df) - churn_count  # Jumlah tidak churn

        # Hitung persentase
        total = len(combined_df)
        yaBaru = round((churn_count / total) * 100, 2)  # Persentase churn
        tidakBaru = round((not_churn_count / total) * 100, 2)  # Persentase tidak churn



        # data yang baru diupload
        # Dapatkan file terbaru berdasarkan waktu modifikasi
        latest_df = max([os.path.join(UPLOAD_FOLDER, f) for f in csv_files], key=os.path.getmtime)

        df_new = pd.read_csv(latest_df)
        #kolom 'churn' yang berisi 1 untuk churn dan 0 untuk tidak churn
        df_new['Churn'] = df_new['Churn'].replace({'YA': 1, 'TIDAK': 0})

        jumlahIya = df_new['Churn'].sum()  # Jumlah churn
        jumlahTidak = len(df_new) - jumlahIya  # Jumlah tidak churn



        # Menghitung classification report dari telkomsel data
        df = pd.read_csv('uploads/0_cth.csv')
        # Menghapus kolom 'ID Pelanggan'
        df.drop(columns=['ID Pelanggan'], inplace=True)
        # mengganti isi kolom Promosi Terakhir dan Churn menjadi 1 dan 0
        Promosi = df.replace(to_replace={'Promosi Terakhir': {'YA': 1, 'TIDAK': 0}})
        Churn = Promosi.replace(to_replace={'Churn': {'YA': 1, 'TIDAK': 0}})
        data2 = Churn

        # menghitung outlier pada tagihan rata-rata
        Q1, Q3 = np.percentile(data2["Tagihan Rata-rata"], [25, 75])
        selisih = Q3 - Q1
        # menghitung batas bawah dan atas
        bw4 = Q1 - 1.5 * selisih
        ba4 = Q3 + 1.5 * selisih

        # Menghitung max dari nilai Tagihan Rata-rata yang bukan outlier
        rerata = data2[(data2["Tagihan Rata-rata"] >= bw4) & (data2["Tagihan Rata-rata"] <= ba4)]["Tagihan Rata-rata"].max()
        # Mengganti outlier dengan max
        data2.loc[(data2["Tagihan Rata-rata"] < bw4) | (data2["Tagihan Rata-rata"] > ba4), "Tagihan Rata-rata"] = rerata

        X_before = data2.drop(['Churn'], axis=1)
        y = data2['Churn']
        # Normalisasi fitur
        X = preprocessing.normalize(X_before)
        # buat fungsi K fold
        def kfold_indices(data, k):
            fold_size = len(data) // k
            indices = np.arange(len(data))
            folds = []
            for i in range(k):
                test_indices = indices[i * fold_size: (i + 1) * fold_size]
                train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
                folds.append((train_indices, test_indices))
            return folds

        # Define the number of folds (K)
        k = 10

        # Get the fold indices
        fold_indices = kfold_indices(X, k)

        # Membuat model RF
        rf = RandomForestClassifier(random_state=45)
        classification_reports = []

        for train_indices, test_indices in fold_indices:
            X_train, y_train = X[train_indices], y[train_indices]
            X_test, y_test = X[test_indices], y[test_indices]
            
            # Melatih model Random Forest
            rf.fit(X_train, y_train)
            
            # Prediksi
            y_pred = rf.predict(X_test)
            
            # Mendapatkan classification report untuk setiap fold
            report = classification_report(y_test, y_pred, output_dict=True)
            classification_reports.append(report)

        # Menghitung rata-rata metrik dari semua fold
        # Placeholder untuk menyimpan hasil rata-rata
        average_report = {}

        # Mendapatkan nama label dan metrik dari classification report pertama
        labels = [label for label in classification_reports[0].keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]
        metrics = classification_reports[0][list(labels)[0]].keys()

        # Menghitung rata-rata setiap metrik untuk setiap label
        for label in labels:
            average_report[label] = {}
            for metric in metrics:
                # Menghitung rata-rata setiap metrik untuk setiap label
                avg_value = np.mean([report[label][metric] for report in classification_reports])
                average_report[label][metric] = round(avg_value, 4)

        # Menghitung rata-rata untuk accuracy
        average_report['accuracy'] = np.mean([report['accuracy'] for report in classification_reports])

        return render_template('comparison.html', df=combined_df, ya_baru=yaBaru, tidak_baru=tidakBaru, ya_lama=yaLama, tidak_lama=tidakLama, jumlah_iya=jumlahIya, jumlah_tidak=jumlahTidak, average_report=average_report)
    else:
        return render_template('comparison.html', message="Belum ada data.", df=None)

# Jalankan aplikasi
if __name__ == '__main__':
    app.run(debug=True)