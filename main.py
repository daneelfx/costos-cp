import os
import pandas as pd
import xml.etree.ElementTree as ET
import logging


def setup_logger(log_file="log.log"):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.handlers = []

    logger.addHandler(file_handler)
    return logger


class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = {}
        self.money = 0

    def add_child(self, child_value):
        if child_value not in self.children:
            self.children[child_value] = TreeNode(child_value)

    def get_child(self, child_value):
        return self.children.get(child_value)

    def calculate_total_money(self):
        total = self.money
        for child in self.children.values():
            total += child.calculate_total_money()
        return total


class Tree:
    def __init__(self):
        self.root = TreeNode("root")

    def insert(self, full_value, money, allowed_nodes=None):
        if '-' in full_value:
            prefix, number = full_value.split('-')
        else:
            prefix, number = None, full_value

        current = self.root

        if prefix:
            if not current.get_child(prefix):
                current.add_child(prefix)
            current = current.get_child(prefix)

        accumulated = ""
        for i in range(1, len(number) + 1):
            accumulated = number[:i]

            if allowed_nodes is None or accumulated in allowed_nodes:
                if not current.get_child(accumulated):
                    current.add_child(accumulated)
                current = current.get_child(accumulated)

        current.money += money

    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root

        for child in node.children.values():
            total_money = child.calculate_total_money()
            print(" " * (level * 2) + f"{child.value}: ${total_money}")
            self.print_tree(child, level + 1)

    def get_rows(self, cost_center=None, node=None, level=0):

        rows = []

        if node is None:
            node = self.root

        for child in node.children.values():
            total_money = child.calculate_total_money()
            if level == 0:
                cost_center = child.value
            if level > 0:
                rows.append((cost_center, child.value, total_money))
            rows.extend(self.get_rows(cost_center, child, level + 1))

        return rows


def parse_xml_to_dataframe(filepath):
    tree = ET.parse(filepath)
    root = tree.getroot()
    records = []

    for record in root.findall('RECORD/ROW'):
        record_data = record.attrib
        records.append(record_data)

    df = pd.DataFrame(records)
    return df


def filter_by_columns(df, column_names):
    df = df[column_names]
    return df


def filter_by_accounts(df, first, second, third):
    filtered_df = df[(df['CUENTA'].str.startswith(first)) | (
        df['CUENTA'].str.startswith(second)) | (df['CUENTA'].str.startswith(third))]
    return filtered_df


def get_value(df):
    if df['CUENTA'].strip().startswith('4'):
        return -df['VALOR'] if df['NATURALEZA'].strip() == 'D' else df['VALOR']
    else:
        return df['VALOR'] if df['NATURALEZA'].strip() == 'D' else -df['VALOR']


def run():
    logger = setup_logger()
    logger.info('*' * 10 + ' EJECUCIÓN INICIADA ' + '*' * 10)
    puc_dataframe = None
    cost_dataframe_result = None
    costs_dataframes = []

    base_path = os.path.dirname(__file__)

    xml_file_paths = []

    for file_name in os.listdir(base_path):
        file_path = os.path.join(base_path, file_name)
        _, file_ext = os.path.splitext(file_path)
        if os.path.isfile(file_path) and file_ext.lower() == '.xml':
            xml_file_paths.append(file_name)

    try:
        for file_path in xml_file_paths:
            df = parse_xml_to_dataframe(file_path)
            df.columns = [column_name.strip().upper()
                          for column_name in df.columns]
            column_names = '_'.join(df.columns)
            if 'CUENTA' in column_names and 'NOMBRE' in column_names and 'NIIF' in column_names:
                df = filter_by_columns(df, ['CUENTA', 'NOMBRE'])
                puc_dataframe = filter_by_accounts(df, '4', '5', '7')
            elif 'FECHA' in column_names and 'CUENTA' in column_names and 'VALOR' in column_names and 'NATURALEZA' in column_names and 'CENTRO' in column_names:
                df = filter_by_columns(
                    df, ['FECHA', 'CUENTA', 'VALOR', 'NATURALEZA', 'CENTRO'])
                df = filter_by_accounts(df, '4', '5', '7')
                df['FECHA'] = pd.to_datetime(
                    df['FECHA'], format='%d/%m/%Y').dt.strftime('%Y%m')
                df['VALOR'] = df['VALOR'].astype(float)
                df['VALOR'] = df.apply(get_value, axis=1)
                del df['NATURALEZA']
                costs_dataframes.append(df)
        logger.info('LOS ARCHIVOS HAN SIDO LEIDOS CON ÉXITO')
    except:
        logger.error('UNO O MÁS ARCHIVOS NO CUMPLEN CON EL FORMATO REQUERIDO')

    result_dataframe = pd.DataFrame(columns=('CENTRO_COSTO', 'CUENTA'))

    try:
        cost_dataframe_result = pd.concat(costs_dataframes, axis=0)
        cost_dataframe_result_copy = cost_dataframe_result.copy()
        cost_dataframe_result_copy = cost_dataframe_result_copy.groupby(
            by=['CENTRO', 'CUENTA', 'FECHA']).sum().reset_index()

        for date in cost_dataframe_result_copy['FECHA'].unique():

            cost_dataframe_date = cost_dataframe_result_copy[cost_dataframe_result_copy['FECHA'] == date].copy(
            )

            cost_dataframe_date['CENTRO_CUENTA'] = cost_dataframe_date['CENTRO'] + '-' + \
                cost_dataframe_date['CUENTA']

            tree = Tree()
            allowed_nodes = set(puc_dataframe['CUENTA'])

            for value, money in cost_dataframe_date[[
                    'CENTRO_CUENTA', 'VALOR']].set_index('CENTRO_CUENTA')['VALOR'].to_dict().items():
                tree.insert(value, money, allowed_nodes)

            current_year_dataframe = pd.DataFrame(
                tree.get_rows(), columns=('CENTRO_COSTO', 'CUENTA', date))
            result_dataframe = pd.merge(
                result_dataframe, current_year_dataframe, how='outer', on=('CENTRO_COSTO', 'CUENTA'))

        result_dataframe = result_dataframe.fillna(value='')
        main_column_names = ['CENTRO_COSTO', 'CUENTA', 'NOMBRE']
        main_column_names += [
            column_name for column_name in result_dataframe.columns if column_name not in main_column_names]

        result_dataframe = pd.merge(
            result_dataframe, puc_dataframe, how='left', on='CUENTA')

        result_dataframe[main_column_names].to_excel(
            './resultado.xlsx', index=False)

        logger.info(
            "LAS TRANSFORMACIONES HAN SIDO REALIZADAS CON ÉXITO. REVISE EL ARCHIVO 'resultado.xlsx'")
    except:
        logger.error(
            'OCURRIÓ UN ERROR AL REALIZAR LAS TRANSFORMACIONES NECESARIAS')


if __name__ == '__main__':
    run()
