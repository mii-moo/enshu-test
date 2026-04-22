import streamlit as st
import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np

st.title("ERP算出ツール")

# --- 画面の状態を管理するフラグ ---
if "step" not in st.session_state:
    st.session_state.step = 1

# --- 「前のステップへ戻る」共通ボタン関数 ---
def back_button(prev_step, reset_keys=None):
    if st.button("← 前のステップへ戻る"):
        if reset_keys:
            for key in reset_keys:
                if key in st.session_state:
                    del st.session_state[key]
        st.session_state.step = prev_step
        st.rerun()

# STEP1----------------------------------------------------------------------------------------------------
if st.session_state.step == 1:
    
    #【表示】全体の説明
    st.markdown("今手元にあるデータは、オドボール課題中の **脳波(EEG)** です")
    st.markdown("これを刺激前後で区切って加算平均することで、 **事象関連電位(ERP)** に変換しましょう")
    st.markdown("")
    st.markdown("")
    st.markdown("具体的には以下のステップで進めます")
    st.markdown("　①データの読み込み")
    st.markdown("　②波形の確認")
    st.markdown("　③エポッキング準備")
    st.markdown("　④エポッキング")
    st.markdown("　⑤ノイズエポック除去")
    st.markdown("　⑥加算平均")
    st.write("---")
    
    #【処理】進む
    if st.button("①データの読み込みへ進む"):
        st.session_state.step = 2
        st.rerun()
# ---------------------------------------------------------------------------------------------------------

# STEP2----------------------------------------------------------------------------------------------------
elif st.session_state.step == 2:
    st.subheader("①データの読み込み")
    st.markdown("")
    
    #【処理】戻る
    back_button(prev_step=1)
    
    #【表示】ファイルアップフォームを表示
    st.markdown("「Upload」から脳波データ（例: sub01_28.CSV）を読み込んでください")
    uploaded_file = st.file_uploader("CSVのアップロードフォーム",type="csv",label_visibility="collapsed")

    #【処理】データがアップされたら
    if uploaded_file is not None:
        try:
            #【処理】元のcsvの名前を覚えておく
            st.session_state.uploaded_filename = uploaded_file.name
            base_name = uploaded_file.name.rsplit(".", 1)[0]
            st.session_state.base_filename = base_name
            
            #【処理】データを読み込み、スペースを取り除く
            df = pd.read_csv(uploaded_file, sep='\t')
            df.columns = (df.columns.str.replace(" ", "").str.strip())
            st.session_state.df = df
            
            #【表示】データの先頭を表示する
            st.success("データの読み込みに成功しました！")
            st.markdown("")
            st.markdown("")
            st.markdown("")
            st.markdown("読み込んだデータの先頭５行を表示しています")
            st.dataframe(df.head())

            #【処理】MNEに変換する
            ch_names = ['Fz', 'Cz', 'Pz', 'EOG', 'S1', 'S2']
            ch_types = ['eeg', 'eeg', 'eeg', 'eog', 'stim', 'stim']
            sfreq = 500  
            info = mne.create_info(ch_names=ch_names,sfreq=sfreq,ch_types=ch_types)
            raw_data = df[ch_names].values.T / 1e6
            raw = mne.io.RawArray(raw_data, info)
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            st.session_state.raw = raw
            st.write("---")

            #【処理】進む
            if st.button("②波形の確認へ進む"):
                st.session_state.step = 3
                st.rerun()

        #【処理】念の為のエラー処理
        except KeyError as e:
            st.error(f"エラー：CSVの中に列名 {e} が見つかりません。")
        except Exception as e:
            st.error(f"予期しないエラーが発生しました: {e}")
# ---------------------------------------------------------------------------------------------------------

# STEP3----------------------------------------------------------------------------------------------------
elif st.session_state.step == 3:
    st.subheader("②波形の確認")

    #【処理】戻る
    back_button(prev_step=2)
    
    #【処理】mneデータを描画するための前処理
    raw = st.session_state.raw
    total_duration = raw.times[-1]
    fig = raw.plot(duration=total_duration,n_channels=len(raw.ch_names),scalings={'eeg': 50e-6, 'eog': 50e-6, 'stim': 1},show_scrollbars=False,show=False)
    fig.set_size_inches(20, 6) 
    
    #【表示】全体の波形とそれの説明を出す
    st.markdown("図にカーソルを合わせると出てくるポップから、表示を大きくできます")
    st.pyplot(fig)
    st.markdown("上から、正中前頭部(Fz)、正中中心部(Cz)、正中頭頂部(Pz)、眼電図(EOG)、標的刺激(S1)、標準刺激(S2)となっているはずです")
    st.write("---")
    
    #【処理】次へ進む
    if st.button("③エポッキング準備に進む"):
        st.session_state.step = 4
        st.rerun()
# ---------------------------------------------------------------------------------------------------------

# STEP4----------------------------------------------------------------------------------------------------
elif st.session_state.step == 4:
    st.subheader("③エポッキング準備")

    #【処理】戻る　そのときステップ内で設定した値はリセットする
    back_button(prev_step=3, reset_keys=["events_all", "event_id"])
    
    #【表示】スライダーの使い方
    st.markdown("スライダーを動かして、S1とS2のトリガ（ㄇ）の立ち上がりに線が引かれるようにしてください")
    st.markdown("S1に赤の線、S2にオレンジの線が引かれることを確認してください")
    st.markdown("現在の検出数が　S1 = 40試行 / S2 = 160試行　か　S1 = 100試行 / S2 = 100試行　になるようにしてください")
    
    #【処理】閾値を変化させる
    my_threshold = st.slider("", 0.0, 0.05, 0.03, step=0.001, format="%.3f")
    raw = st.session_state.raw
    raw_stim = raw.copy()
    def binarize_stim(data):
        return (data > my_threshold).astype(float)
    raw_stim.apply_function(binarize_stim, picks=['S1', 'S2'])
    evs_s1 = mne.find_events(raw_stim, stim_channel='S1', output='onset', verbose=False)
    evs_s2 = mne.find_events(raw_stim, stim_channel='S2', output='onset', verbose=False)

    #【処理】何かしら検出された場合
    if len(evs_s1) > 0 or len(evs_s2) > 0:
        #【表示】現在のスライダー位置での検出数を表示する
        st.success(f"現在の検出数： S1 = {len(evs_s1)}試行 / S2 = {len(evs_s2)}試行")
        
        #【処理】S1とS2でそれぞれ閾値のフラグを立てて、統合する
        if len(evs_s2) > 0:
            evs_s2[:, 2] = 2
        events_all = np.concatenate([evs_s1, evs_s2]) if (len(evs_s1) > 0 and len(evs_s2) > 0) else (evs_s1 if len(evs_s1) > 0 else evs_s2)
        events_all = events_all[np.argsort(events_all[:, 0])]

        #【表示】更新後の波形
        event_color = {1: 'red', 2: 'orange'}
        fig = raw_stim.plot(duration=raw_stim.times[-1],n_channels=len(raw_stim.ch_names),scalings={'eeg': 50e-6, 'eog': 50e-6, 'stim': 1},events=events_all,event_color=event_color,event_id=None,show_scrollbars=False,show=False)
        fig.set_size_inches(15, 6)
        st.pyplot(fig)
        st.write("---")

        #【処理】次へ進む
        if st.button("④エポッキングに進む"):
            st.session_state.events_all = events_all
            st.session_state.event_id = {'S1': 1, 'S2': 2}
            st.session_state.step = 5
            st.rerun()

    #【処理】何も検出されなかった場合
    else:
        st.warning("イベントが検出されていません　スライダーを下げてみてください")
# ---------------------------------------------------------------------------------------------------------

# STEP5----------------------------------------------------------------------------------------------------
elif st.session_state.step == 5:
    st.subheader("④エポッキング")

    #【処理】戻る　そのときステップ内で設定した値はリセットする
    back_button(prev_step=4, reset_keys=["epochs", "epoch_idx", "bad_epochs", "all_checked"])

    #【処理】③エポッキング準備で用意したトリガーを使って、エポッキングする
    raw = st.session_state.raw
    events_all = st.session_state.events_all
    event_id = st.session_state.event_id
    tmin, tmax = -0.2, 0.6
    if "epochs" not in st.session_state:
        epochs = mne.Epochs(
            raw, events_all, event_id=event_id,
            tmin=tmin, tmax=tmax, baseline=(None, 0),
            preload=True, verbose=False
        )
        st.session_state.epochs = epochs
        st.session_state.epoch_idx = 0
        st.session_state.bad_epochs = []
    epochs = st.session_state.epochs
    st.success(f"エポッキング完了: {len(epochs)} 試行")
    st.markdown("200試行になっていることを確認してください")

    #【表示】４つの波形を表示し、ノイズエポックの存在を知らせる
    data_uv = epochs.get_data() * 1e6
    times = epochs.times * 1000
    event_codes = epochs.events[:, 2]
    target_chs = ['Fz', 'Cz', 'Pz', 'EOG']
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for ax, ch_name in zip(axes, target_chs):
        ch_idx = epochs.ch_names.index(ch_name)
        for i in range(len(data_uv)):
            c = 'red' if event_codes[i] == 1 else 'orange'
            ax.plot(times, data_uv[i, ch_idx, :], color=c, lw=0.5, alpha=0.3)
        ax.set_title(ch_name)
        ax.axvline(0, color='black', lw=1)
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("全体と大きく外れている波形はありませんか？　ノイズエポックを除去しましょう")
    st.write("---")

    #【処理】次へ進む
    if st.button("⑤ノイズエポック除去に進む"):
        st.session_state.step = 6
        st.rerun()
# ---------------------------------------------------------------------------------------------------------

# STEP6----------------------------------------------------------------------------------------------------
elif st.session_state.step == 6:
    st.subheader("⑤ノイズエポック除去")
    
    epochs = st.session_state.epochs

    if "epochs_original" not in st.session_state:
        st.session_state.epochs_original = epochs.copy()

    if "n_epochs_original" not in st.session_state:
        st.session_state.n_epochs_original = len(epochs)

    #【処理】戻る　そのときステップ内で設定した値はリセットする
    back_button(prev_step=5, reset_keys=["epoch_idx", "epoch_status", "all_checked"])

    #【処理】それぞれのエポックを初期状態として0にする
    # epochs = st.session_state.epochs
    if "epoch_idx" not in st.session_state:
        st.session_state.epoch_idx = 0
    if "epoch_status" not in st.session_state:
        st.session_state.epoch_status = {}
    idx = st.session_state.epoch_idx
    n_epochs = len(epochs)

    #【表示】表示中の試行をスライダーで表示する
    new_idx_display = st.slider(
        "試行番号",
        min_value=1,
        max_value=n_epochs,
        value=idx + 1,
        step=1
    )

    #【表示】現在の試行を判定状況と現在の進捗を表示する
    new_idx = new_idx_display - 1
    if new_idx != idx:
        st.session_state.epoch_idx = new_idx
        st.rerun()
    status = st.session_state.epoch_status.get(idx, 0)
    if status == 1:
        st.success("この試行は採用されています")
    elif status == -1:
        st.error("この試行は棄却されています")
    else:
        st.info("この試行は未判定です")
    n_accept = sum(1 for v in st.session_state.epoch_status.values() if v == 1)
    n_reject = sum(1 for v in st.session_state.epoch_status.values() if v == -1)
    st.write(f"採用: {n_accept} / 棄却: {n_reject} / 未判定: {n_epochs - n_accept - n_reject}")
    
    #【表示】波形を表示する
    data = epochs.get_data()[idx]
    times = epochs.times * 1000
    fig, ax = plt.subplots(figsize=(10, 6))
    target_chs = ['Fz', 'Cz', 'Pz', 'EOG']
    for ch_name in target_chs:
        ch_idx = epochs.ch_names.index(ch_name)
        ax.plot(times, data[ch_idx] * 1e6, label=ch_name)
    ax.axvline(0, color='black', linestyle='--')
    ax.set_ylim(-100, 100)
    ax.set_ylabel("μV")
    ax.set_title(f"Trial {idx + 1} - Stimulus: {'S1' if epochs.events[idx, 2] == 1 else 'S2'}")
    ax.legend(loc='upper right')
    ax.set_xlabel("Time (ms)")
    plt.tight_layout()
    st.pyplot(fig)

    #【表示】３つのボタンを表示する
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("←前の試行に戻る"):
            if idx > 0:
                st.session_state.epoch_idx -= 1
                st.rerun()
    with col2:
        if st.button("採用して次へ"):
            st.session_state.epoch_status[idx] = 1
            if idx < n_epochs - 1:
                st.session_state.epoch_idx += 1
                st.rerun()
            else:
                st.session_state.all_checked = True
    with col3:
        if st.button("棄却して次へ"):
            st.session_state.epoch_status[idx] = -1

            if idx < n_epochs - 1:
                st.session_state.epoch_idx += 1
                st.rerun()
            else:
                st.session_state.all_checked = True
    
    #【表示】全てのチェックが完了したら、完了したことを表示する
    all_done = len(st.session_state.epoch_status) == n_epochs
    if all_done:
        st.success("全試行のチェックが完了しました！")

        bad_epochs = [i for i, v in st.session_state.epoch_status.items() if v == -1]
        epochs_clean = st.session_state.epochs_original.copy()
        #st.session_state.n_epochs_original = len(epochs.events) + len(bad_epochs)
        st.write("---")

        #【処理】次へ進む
        if st.button("⑥加算平均へ進む"):
            epochs_clean.drop(bad_epochs)
            st.session_state.epochs = epochs_clean
            st.session_state.step = 7
            st.rerun()
# ---------------------------------------------------------------------------------------------------------

# STEP7----------------------------------------------------------------------------------------------------
elif st.session_state.step == 7:
    st.subheader("⑥加算平均")

    #【処理】ノイズエポック除去をはじめからやり直す　バグるから
    if st.button("⚠️ノイズエポック除去をはじめからやり直す"):
        st.session_state.step = 6
        st.session_state.epoch_status = {}
        st.session_state.epoch_idx = 0
        st.rerun()
    st.markdown("")

    #【処理】各エポックをS1とS2に分けて平均する
    epochs = st.session_state.epochs
    evoked_s1 = epochs['S1'].average()
    evoked_s2 = epochs['S2'].average()
    
    #【表示】試行数の表示
    #n_total = st.session_state.get("n_epochs_original", len(epochs))
    #n_accepted = len(epochs)
    #n_rejected = n_total - n_accepted
    #st.write(f"採用試行数: {n_accepted}　棄却試行数: {n_rejected}　総試行数: {n_total}")
    #st.markdown("")
    
    epoch_status = st.session_state.get("epoch_status", {})

    n_total = st.session_state.n_epochs_original
    n_accepted = sum(1 for v in epoch_status.values() if v == 1)
    n_rejected = sum(1 for v in epoch_status.values() if v == -1)

    st.write(f"採用: {n_accepted}　棄却: {n_rejected}　総数: {n_total}")

    #【表示】グラフを描画する
    st.markdown("S1（標的刺激）と S2（標準刺激）の加算平均波形")
    target_chs = ['Fz', 'Cz', 'Pz']
    times = evoked_s1.times * 1000 
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, ch_name in zip(axes, target_chs):
        ch_idx = evoked_s1.ch_names.index(ch_name)
        val_s1 = evoked_s1.data[ch_idx] * 1e6
        val_s2 = evoked_s2.data[ch_idx] * 1e6
        ax.plot(times, val_s1, color='red', label='S1', lw=2)
        ax.plot(times, val_s2, color='orange', label='S2', lw=2)
        ax.axvline(0, color='black', lw=1)
        ax.axhline(0, color='black', lw=0.5, alpha=0.5)
        ax.set_title(ch_name)
        ax.set_ylabel("μV")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
    axes[-1].set_xlabel("Time (ms)")
    plt.tight_layout()
    st.pyplot(fig)
    st.markdown("")

    #【処理】csvとして出力するデータを整形する
    times = evoked_s1.times * 1000
    export_df = pd.DataFrame({"Time_ms": times})
    for ch_name in evoked_s1.ch_names:
        idx = evoked_s1.ch_names.index(ch_name)
        export_df[f"{ch_name}_S1"] = evoked_s1.data[idx] * 1e6
        export_df[f"{ch_name}_S2"] = evoked_s2.data[idx] * 1e6

    #【表示】csvデータの先頭を表示する
    st.markdown("加算平均の結果をダウンロードできます")
    st.dataframe(export_df.head())
    
    #【処理】csvとして出力するデータを整形する
    csv = export_df.to_csv(index=False).encode('utf-8')
    base_name = st.session_state.get("base_filename", "result")
    output_name = f"{base_name}_erp.csv"

    #【表示】ダウンロードボタン
    st.download_button(
        label="ダウンロード",
        data=csv,
        file_name=output_name,
        mime="text/csv",
    )
    st.write("---")

    #【処理】一番最初に戻る
    if st.button("最初に戻る"):
        st.session_state.clear()
        st.rerun()
# ---------------------------------------------------------------------------------------------------------