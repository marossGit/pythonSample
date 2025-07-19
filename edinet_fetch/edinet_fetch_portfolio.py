# EDINET API から財務データを取得・解析・キャッシュ処理を行うクラス
# 公開用にコメントを充実させたポートフォリオ向けコード（APIキーやIDは伏せています）

import os
import re
import shutil
import datetime
import pickle
import zipfile
from pathlib import Path
import requests
from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser
import pandas as pd


class EdinetDataFetcher:
    """
    EDINETから提出された有価証券報告書（XBRL形式）を取得・解析・キャッシュ処理するクラス。
    - EDINET APIからzip/XBRLデータを取得
    - xbrlファイルをパースして財務情報を抽出
    - ファイル差分を検出し、キャッシュを無効化
    - ダウンロード済みzipをローカルに保存し、再利用可能に
    """

    def __init__(self):
        self.parser = EdinetXbrlParser()
        self.parsed_data = None

    def get_data_by_keys(self, key_context_list, field_name):
        """候補キーとコンテキストでデータを検索。取得できない場合はNone。"""
        for key, context in key_context_list:
            try:
                d = self.parsed_data.get_data_by_context_ref(key, context)
                return float(d.get_value())
            except Exception:
                continue
        print(f"[WARN] '{field_name}' が見つかりませんでした。")
        return None

    def parse_xbrl_file(self, file_path):
        self.parsed_data = self.parser.parse_file(file_path)

    def extract_financial_data(self, xbrl_path, stock_obj):
        """
        指定されたXBRLファイルから財務情報を抽出し、1行のDataFrameとして返す。
        """
        self.parse_xbrl_file(xbrl_path)

        data = {
            'endDate': re.findall(r"\d{4}-\d{2}-\d{2}", str(xbrl_path))[-1],
            'netSales': self.get_data_by_keys([
                ("jppfs_cor:NetSales", "CurrentYearDuration"),
                ("jpigp_cor:netsalesifrs", "CurrentYearDuration")
            ], "売上高"),
            'OperatingIncome': self.get_data_by_keys([
                ("jppfs_cor:OperatingIncome", "CurrentYearDuration"),
                ("jpigp_cor:operatingprofitlossifrs", "CurrentYearDuration")
            ], "営業利益"),
            'NetIncomeLoss': self.get_data_by_keys([
                ("jppfs_cor:ProfitLoss", "CurrentYearDuration"),
                ("jpigp_cor:profitlossbeforetaxifrs", "CurrentYearDuration")
            ], "純利益")
        }

        # 年度平均株価も取得（株価データは外部から事前に与えられる前提）
        try:
            year = int(data['endDate'][:4])
            avg_close = stock_obj.dayData.loc[str(year), "Close"].mean()
            data["avgClose"] = avg_close
        except:
            data["avgClose"] = stock_obj.dayData["Close"].iloc[-1]

        return pd.DataFrame([data])

    def extract_all_from_folder(self, stock_obj):
        """
        ローカルのzipファイル（EDINETからDL済）を展開し、すべてのXBRLから財務情報を抽出。
        キャッシュとzipファイルの差分も検知し、変化があればキャッシュ削除。
        """
        base_dir = Path(__file__).resolve().parent / "annual_data"
        target_folder = [f for f in base_dir.glob(f"*_{stock_obj.companyId}") if f.is_dir()]
        if not target_folder:
            print("対象フォルダが見つかりません。")
            return pd.DataFrame()
        folder = target_folder[0]

        # キャッシュ差分検出
        zip_files = sorted([f.name for f in folder.glob("*S.zip")])
        record_path = folder / "zip_record.pkl"
        old_list = []
        if record_path.exists():
            try:
                with open(record_path, 'rb') as f:
                    old_list = pickle.load(f)
            except:
                pass
        if set(zip_files) != set(old_list):
            cache_path = folder / f"{stock_obj.companyId}_financial_data.pkl"
            if cache_path.exists():
                cache_path.unlink()  # 差分があればキャッシュ削除
            with open(record_path, 'wb') as f:
                pickle.dump(zip_files, f)

        # キャッシュがあれば使用
        cache_path = folder / f"{stock_obj.companyId}_financial_data.pkl"
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    print("キャッシュを使用：", cache_path.name)
                    return pickle.load(f)
            except:
                pass

        # zip展開と解析
        temp_path = folder / "temp"
        if temp_path.exists():
            shutil.rmtree(temp_path)
        temp_path.mkdir(exist_ok=True)

        all_df = pd.DataFrame()
        for zip_file in folder.glob("*S.zip"):
            try:
                with zipfile.ZipFile(zip_file) as zf:
                    zf.extractall(temp_path)
            except:
                continue

        for xbrl in temp_path.rglob("*.xbrl"):
            if "AuditDoc" in xbrl.parts:
                continue
            try:
                df = self.extract_financial_data(xbrl, stock_obj)
                all_df = pd.concat([all_df, df], ignore_index=True)
            except:
                continue

        shutil.rmtree(temp_path)

        with open(cache_path, 'wb') as f:
            pickle.dump(all_df, f)

        return all_df
