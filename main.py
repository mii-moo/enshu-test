import streamlit as st
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np

st.title("ERP解析演習：マニュアル")

# --- 画面の状態を管理するフラグ ---
if "step" not in st.session_state:
    st.session_state.step = 1

# --- ①データの読み込みセクション ---
if st.session_state.step == 1:
    st.subheader("①データの読み込み")
    uploaded_file = st.file_uploader("CSVファイルを選択してください。", type="csv")

    if uploaded_file is not None:
        # 読み込み
        df = pd.read_csv(uploaded_file, sep='\t')
        st.session_state.df = df # 先に保存しておく
        
        st.write("データの先頭5行を表示しています。")
        st.dataframe(df.head())
        
        st.info("現時点のチャンネル名には不要なスペースや分かりにくい名前が含まれています。")
        
        if st.button("チャンネル名変更へ進む"):
            st.session_state.step = 2
            st.rerun()

# --- ②チャンネル名の変更セクション ---
elif st.session_state.step == 2:
    st.subheader("②チャンネル名の変更")
    df = st.session_state.df

    # --- 自動置換ボタンの設置（これがあると楽です！） ---
    if st.button("推奨されるルールで自動置換する"):
        # スペース削除、EXT -> S1, EXT.1 -> S2 のルールを適用
        new_cols = [c.strip().replace("EXT.1", "S2").replace("EXT", "S1") for c in df.columns]
        df.columns = new_cols
        st.session_state.df = df
        st.rerun()

    st.write("手動で修正が必要な場合は以下を書き換えてください。")
    
    input_cols = st.columns(4)
    new_names = []
    
    for i, old_name in enumerate(df.columns):
        with input_cols[i % 4]:
            user_input = st.text_input(f"列 {i+1}", value=old_name, key=f"input_{i}")
            new_names.append(user_input)
    
    if st.button("確定して次へ"):
        df.columns = new_names
        st.session_state.df = df 
        st.session_state.step = 3
        st.rerun()

# --- ③チャンネル名の確認セクション ---
elif st.session_state.step == 3:
    st.subheader("③変更結果の確認")
    df = st.session_state.df
    
    st.success("チャンネル名を変更しました。")
    st.dataframe(df.head())
    
    st.write("これを脳波データ（MNEオブジェクト）として扱うための形式に変えましょう。")
    
    if st.button("MNEデータ形式へ変換"):
        st.session_state.step = 4
        st.rerun()

elif st.session_state.step == 4:
    st.subheader("④MNE形式への変換と電極設定")
    df = st.session_state.df
    
    # 変換済みかどうかのフラグを初期化
    if "is_converted" not in st.session_state:
        st.session_state.is_converted = False

    ch_names = ['Fz', 'Cz', 'Pz', 'EOG', 'S1', 'S2']
    ch_types = ['eeg', 'eeg', 'eeg', 'eog', 'stim', 'stim']
    sfreq = 500
    
    st.write("以下の設定でMNEデータを作成します：")
    st.write(f"- チャンネル: {', '.join(ch_names)}")
    st.write(f"- サンプリング周波数: {sfreq} Hz")

    # --- ボタン1：変換を実行 ---
    if st.button("変換を実行"):
        try:
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
            raw_data = df[ch_names].values.T / 1e6
            raw = mne.io.RawArray(raw_data, info)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            
            st.session_state.raw = raw
            st.session_state.is_converted = True # 変換成功フラグを立てる
            
        except KeyError as e:
            st.error(f"エラー：CSVの中に列名 {e} が見つかりません。")

    # --- 変換が終わっていたら表示するセクション ---
    if st.session_state.is_converted:
        st.success("MNE形式への変換が完了しました！")
        
        st.write("### 電極配置の確認")
        temp_raw = st.session_state.raw.copy().pick_types(eeg=True)
        fig = temp_raw.plot_sensors(show_names=True, show=False)
        st.pyplot(fig)
        
        # ステップ移動ボタンは「変換ボタン」の外に出す！
        if st.button("次へ（波形確認）"):
            st.session_state.step = 5
            # 次のステップへ行くのでフラグをリセットしておく（任意）
            st.session_state.is_converted = False 
            st.rerun()
# --- ⑤波形の確認セクション ---
elif st.session_state.step == 5:
    st.subheader("⑤全体波形の確認")
    raw = st.session_state.raw

    # 1. 全体時間を計算
    total_duration = raw.times[-1]
    
    # 2. MNEのプロット機能で図を作成
    # scalings は MNEの内部単位（V）に対して指定するため、50μVなら 50e-6 で正解です！
    fig = raw.plot(
        duration=total_duration,
        n_channels=len(raw.ch_names),
        scalings={'eeg': 50e-6, 'eog': 50e-6, 'stim': 1}, # stimはそのまま
        show_scrollbars=False,
        show=False
    )

    # 3. サイズの調整
    fig.set_size_inches(20, 6) 
    
    # 4. Streamlitへの表示
    st.pyplot(fig)

    st.info(f"データの全期間（{total_duration:.1f} 秒）を表示しています。")
    
    if st.button("次へ（前処理：フィルタリング）"):
        st.session_state.step = 6
        st.rerun()

elif st.session_state.step == 6:
    st.subheader("⑥トリガー（イベント）の抽出（リアルタイム設定）")
    raw = st.session_state.raw

    st.write("スライダーを動かして、S1/S2の立ち上がりに正しく線が重なるように調整してください。")

    # 1. スライダー（これ自体がトリガーになり、動かすたびに再実行されます）
    my_threshold = st.slider("二値化のしきい値（V）", 0.0, 0.1, 0.05, step=0.005, format="%.3f")

    # 2. 抽出処理（ボタンなしで毎回実行）
    raw_stim = raw.copy()
    def binarize_stim(data):
        return (data > my_threshold).astype(float)

    raw_stim.apply_function(binarize_stim, picks=['S1', 'S2'])

    evs_s1 = mne.find_events(raw_stim, stim_channel='S1', output='onset', verbose=False)
    evs_s2 = mne.find_events(raw_stim, stim_channel='S2', output='onset', verbose=False)

    # 3. 結果の表示
    if len(evs_s1) > 0 or len(evs_s2) > 0:
        st.success(f"現在の検出数： S1 = {len(evs_s1)}回 / S2 = {len(evs_s2)}回")
        
        # IDの付与と統合
        if len(evs_s2) > 0:
            evs_s2[:, 2] = 2
        
        # 片方しか取れていない場合も考慮
        events_all = np.concatenate([evs_s1, evs_s2]) if (len(evs_s1) > 0 and len(evs_s2) > 0) else (evs_s1 if len(evs_s1) > 0 else evs_s2)
        events_all = events_all[np.argsort(events_all[:, 0])]

        # 波形のプロット
        event_color = {1: 'red', 2: 'orange'}
        fig = raw_stim.plot(
            duration=raw_stim.times[-1],
            n_channels=len(raw_stim.ch_names),
            scalings={'eeg': 50e-6, 'eog': 50e-6, 'stim': 1},
            events=events_all,
            event_color=event_color,
            show_scrollbars=False,
            show=False
        )
        fig.set_size_inches(15, 6)
        st.pyplot(fig)

        # 4. 「次へ」ボタン
        # このボタンを押した時だけデータを保存してステップを進める
        if st.button("このしきい値で確定してエポッキングへ"):
            st.session_state.events_all = events_all
            st.session_state.event_id = {'S1': 1, 'S2': 2}
            st.session_state.step = 7
            st.rerun()
    else:
        st.warning("イベントが検出されていません。しきい値を下げてみてください。")

elif st.session_state.step == 7:
    st.subheader("⑦エポッキングと目視チェック")
    raw = st.session_state.raw
    events_all = st.session_state.events_all
    event_id = st.session_state.event_id

    # エポック作成済みフラグ
    if "is_epoched" not in st.session_state:
        st.session_state.is_epoched = False

    tmin, tmax = -0.2, 0.6

    # --- ボタン：エポッキングを実行 ---
    if st.button("エポッキングを実行"):
        epochs = mne.Epochs(
            raw, events_all, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=(None, 0),
            preload=True, verbose=False
        )
        st.session_state.epochs = epochs
        st.session_state.is_epoched = True
        # 目視チェック用のインデックス等も初期化
        st.session_state.epoch_idx = 0
        st.session_state.bad_epochs = []

    # --- エポック作成後の表示 ---
    if st.session_state.is_epoched:
        epochs = st.session_state.epochs
        st.success(f"エポッキング完了: {len(epochs)} 試行")

        # 重ね描きのプレビュー（確認用なので少し小さめでもOK）
        with st.expander("全試行の重ね描きを確認する"):
            data_uv = epochs.get_data() * 1e6
            times = epochs.times * 1000
            event_codes = epochs.events[:, 2]
            target_chs = ['Fz', 'Cz', 'Pz']
            
            fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
            for ax, ch_name in zip(axes, target_chs):
                ch_idx = epochs.ch_names.index(ch_name)
                for i in range(len(data_uv)):
                    c = 'red' if event_codes[i] == 1 else 'orange'
                    ax.plot(times, data_uv[i, ch_idx, :], color=c, lw=0.5, alpha=0.3)
                ax.set_title(ch_name)
                ax.axvline(0, color='black', lw=1)
            plt.tight_layout()
            st.pyplot(fig)

        st.write("---")
        st.write("### 次のステップ：1試行ずつの目視チェック")
        st.write("EVENTチャンネルの不備記録や、脳波の大きなノイズを確認して採用・棄却を決定します。")
        
        # ここでボタンを外に出す
        if st.button("目視チェックを開始する"):
            st.session_state.step = 8 # 目視チェック専用のサブステップへ
            st.rerun()

elif st.session_state.step == 8:
    st.subheader("⑧：試行ごとの手動チェック（不備記録確認）")
    epochs = st.session_state.epochs
    
    if "epoch_idx" not in st.session_state:
        st.session_state.epoch_idx = 0
    if "bad_epochs" not in st.session_state:
        st.session_state.bad_epochs = []

    idx = st.session_state.epoch_idx
    n_epochs = len(epochs)

    st.write(f"試行 {idx + 1} / {n_epochs}")
    st.progress((idx + 1) / n_epochs)

    # 1. データの準備
    data = epochs.get_data()[idx]
    times = epochs.times * 1000
    
    # 2. 描画（2段構成：上が脳波、下が不備トリガー）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})

    # --- 上段：脳波 (EEG) ---
    for ch_idx, ch_name in enumerate(epochs.ch_names):
        if ch_name in ['Fz', 'Cz', 'Pz']:
            ax1.plot(times, data[ch_idx] * 1e6, label=ch_name)
    
    ax1.axvline(0, color='black', linestyle='--')
    ax1.set_ylim(-100, 100)
    ax1.set_ylabel("EEG (μV)")
    ax1.set_title(f"Trial {idx + 1} - Stimulus: {'S1' if epochs.events[idx, 2] == 1 else 'S2'}")
    ax1.legend(loc='upper right')

    # --- 下段：不備記録 (EVENT) ---
    if 'EVENT' in epochs.ch_names:
        event_idx = epochs.ch_names.index('EVENT')
        ax2.plot(times, data[event_idx], color='purple', label='Error Trigger')
        ax2.set_ylabel("EVENT")
        ax2.set_ylim(-0.1, 1.1) # 二値化されている想定
        ax2.fill_between(times, 0, data[event_idx], color='purple', alpha=0.2)
        ax2.legend(loc='upper right')
    else:
        ax2.text(0.5, 0.5, "EVENT channel not found", ha='center')

    ax2.set_xlabel("Time (ms)")
    plt.tight_layout()
    st.pyplot(fig)

    # 3. 注意喚起のメッセージ
    if 'EVENT' in epochs.ch_names and np.any(data[event_idx] > 0.5):
        st.warning("⚠️ この試行中、EVENTチャンネルに不備記録の反応があります。棄却を検討してください。")

    # 4. 操作ボタン（前回同様）
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ 採用して次へ"):
            if idx < n_epochs - 1:
                st.session_state.epoch_idx += 1
                st.rerun()
            else:
                st.session_state.all_checked = True # 完了フラグ

    with col2:
        if st.button("❌ 棄却して次へ"):
            if idx not in st.session_state.bad_epochs:
                st.session_state.bad_epochs.append(idx)
            if idx < n_epochs - 1:
                st.session_state.epoch_idx += 1
                st.rerun()
            else:
                st.session_state.all_checked = True # 完了フラグ
    
    st.write("---")
    # 開発用メニュー（サイドバーに入れるか、目立たない場所に配置）
    with st.expander("🛠️ 開発用ショートカット"):
        if st.button("残りすべての試行を「採用」にする"):
            # すべてチェック済みフラグを立てて、最後のインデックスへ飛ばす
            st.session_state.epoch_idx = n_epochs - 1
            st.session_state.all_checked = True
            st.rerun()
            
        if st.button("全棄却（テスト用）"):
            st.session_state.bad_epochs = list(range(n_epochs))
            st.session_state.all_checked = True
            st.rerun()

    # 5. 全試行の確認が終わったら確定ボタンを出す
    if st.session_state.get("all_checked", False):
        st.write("---")
        st.success("すべての試行のチェックが完了しました！")
        st.info(f"棄却する試行数: {len(st.session_state.bad_epochs)} / 総試行数: {n_epochs}")
        if st.button("チェックを確定して加算平均へ進む"):
            # 棄却実行
            epochs.drop(st.session_state.bad_epochs)
            st.session_state.epochs = epochs
            st.session_state.step = 9
            st.rerun()

elif st.session_state.step == 9:
    st.subheader("⑨加算平均（ERPの算出と比較）")
    epochs = st.session_state.epochs

    # 1. 各条件の平均（Evoked）を算出
    evoked_s1 = epochs['S1'].average()
    evoked_s2 = epochs['S2'].average()

    st.write("S1（刺激）と S2（反応）の平均波形を比較します。")

    # 2. チャンネル選択
    target_ch = st.selectbox("表示するチャンネルを選択してください", ['Pz', 'Cz', 'Fz'])

    # 3. グラフの描画
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # タイム軸をmsに変換
    times = evoked_s1.times * 1000 
    
    # 各データの取得（uV単位）
    ch_idx = evoked_s1.ch_names.index(target_ch)
    val_s1 = evoked_s1.data[ch_idx] * 1e6
    val_s2 = evoked_s2.data[ch_idx] * 1e6

    ax.plot(times, val_s1, color='red', label='S1 (Stimulus)', lw=2)
    ax.plot(times, val_s2, color='orange', label='S2 (Response)', lw=2)

    # 装飾
    ax.axvline(0, color='black', lw=1) # 刺激時点
    ax.axhline(0, color='black', lw=0.5, alpha=0.5)
    ax.set_title(f"ERP Comparison at {target_ch}")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude (μV)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 脳波の慣習に合わせて、y軸のプラスマイナスを反転させるオプション
    if st.checkbox("陰性を上向きに表示（脳波の伝統的表示）"):
        ax.invert_yaxis()

    st.pyplot(fig)
    
    # 1. 保存用データの作成
    # 時間軸、S1の平均、S2の平均をまとめたDataFrameを作る
    times = evoked_s1.times * 1000  # ms
    
    # 全チャンネルのデータを一度に保存するための準備
    export_df = pd.DataFrame({"Time_ms": times})
    
    for ch_name in evoked_s1.ch_names:
        idx = evoked_s1.ch_names.index(ch_name)
        export_df[f"{ch_name}_S1_avg"] = evoked_s1.data[idx] * 1e6
        export_df[f"{ch_name}_S2_avg"] = evoked_s2.data[idx] * 1e6

    st.write("各チャンネルの加算平均値（μV）をCSVとしてダウンロードできます。")
    st.dataframe(export_df.head())

    # 2. Streamlitのダウンロードボタン
    csv = export_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="解析結果をCSVでダウンロード",
        data=csv,
        file_name="erp_result.csv",
        mime="text/csv",
    )

    if st.button("最初に戻る"):
        # セッションをクリアしてステップ1へ 
        st.session_state.clear()
        st.rerun()