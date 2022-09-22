# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import math
import datetime
import numpy as np
import pandas as pd
from collections import namedtuple
from py2neo import Graph, Node, Relationship
import logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename='output.log',
                    level=logging.INFO)
logger = logging.getLogger('')


def extract_matching_pairs(df, amount_threshold, allowed_amountdiff, allowed_timediff, trx_info):
    pairs = []
    for (_, trxid_i, sacctid_i, racctid_i, amt_i, time_i) in df.itertuples():
        for (_, trxid_j, sacctid_j, racctid_j, amt_j, time_j) in df.itertuples():
            if trxid_i == trxid_j:
                continue
            amountdiff = abs(amt_i - amt_j)
            timediff = abs(time_i - time_j)

            if ((sacctid_j == racctid_i) &
                    (amt_i >= amount_threshold) &
                    (amountdiff <= allowed_amountdiff) &
                    (timediff <= allowed_timediff)):

                pair_found = (trx_info(trxid_i, sacctid_i, racctid_i, amt_i, time_i),
                              trx_info(trxid_j, sacctid_j, racctid_j, amt_j, time_j),
                              1/(amountdiff * timediff))
                pairs.append(pair_found)
    return pairs


def create_node_relationships(graph, pairs):
    node_vi = node_vj = node_vk = Node()
    rel_eij = rel_ejk = Relationship(node_vi)
    n_data_out = "MATCH (n:account {{id: {}}}) RETURN n LIMIT 1"
    r_data_out = "MATCH p=(:account {{id: {}}})-[r:send]->(:account {{id: {}}}) RETURN p LIMIT 1"
    r_data_in = "MATCH p=(:account {{id: {}}})-[r:send]->(:account {{id: {}}}) SET r.weight={}"

    for pair in pairs:

        # create nodes if they don't exist
        n_query = n_data_out.format(pair[0].sacctid)
        if not graph.evaluate(n_query):
            node_vi = Node("account", id=pair[0].sacctid)
            graph.create(node_vi)
            logger.info(f"Created node {node_vi['id']} ....")

        n_query = n_data_out.format(pair[0].racctid)
        if not graph.evaluate(n_query):
            node_vj = Node("account", id=pair[0].racctid)
            graph.create(node_vj)
            logger.info(f"Created node {node_vj['id']} ")

        n_query = n_data_out.format(pair[1].racctid)
        if not graph.evaluate(n_query):
            node_vk = Node("account", id=pair[1].racctid)
            graph.create(node_vk)
            logger.info(f"Created node {node_vk['id']} ")

        # create relationships if they don't exist, increment weights if they do exist
        # pair 1
        r_query_out = r_data_out.format(pair[0].sacctid, pair[0].racctid)
        rel_data = graph.evaluate(r_query_out)
        if not rel_data:
            rel_eij = Relationship(node_vi, "send", node_vj, weight=1, amount=pair[0].amt, time=pair[0].time,
                                   id=pair[0].trxid)
            graph.create(rel_eij)
            logger.info(f"Added relationship {node_vi['id']} --> {node_vj['id']} with rel_id {rel_eij['id']} ")
        else:
            r_query_in = r_data_in.format(pair[0].sacctid, pair[0].racctid, (rel_data.relationships[0]['weight']) + 1)
            graph.evaluate(r_query_in)
            logger.info(f"Relationship {node_vi['id']} --> {node_vj['id']} already exists. "
                        f"Increment weight {rel_eij['id']} ....")

        # pair 2
        r_query_out = r_data_out.format(pair[1].sacctid, pair[1].racctid)
        rel_data = graph.evaluate(r_query_out)
        if not rel_data:
            rel_ejk = Relationship(node_vj, "send", node_vk, weight=1, amount=pair[1].amt, time=pair[1].time,
                                   id=pair[1].trxid)
            graph.create(rel_ejk)
            logger.info(f"Added relationship {node_vj['id']} --> {node_vk['id']} with rel_id {rel_ejk['id']} ")
        else:
            r_query = r_data_in.format(pair[1].sacctid, pair[1].racctid, (rel_data.relationships[0]['weight']) + 1)
            graph.evaluate(r_query)
            logger.info(f"Relationship {node_vj['id']} --> {node_vk['id']} already exists. "
                        f"Increment weight {rel_ejk['id']} ....")


def calculate_total_weights(graph):
    for rel in graph.match(r_type="send"):
        if rel.nodes[1]['id'] not in incoming_rels:
            incoming_rels.update({rel.nodes[1]['id']: 1})
            incoming_weights.update({rel.nodes[1]['id']: rel['weight']})
        else:
            incoming_rels.update({rel.nodes[1]['id']: incoming_rels[rel.nodes[1]['id']]+1})
            incoming_weights.update({rel.nodes[1]['id']: incoming_weights[rel.nodes[1]['id']]+rel['weight']})

        if rel.nodes[0]['id'] not in outgoing_rels:
            outgoing_rels.update({rel.nodes[0]['id']: 1})
            outgoing_weights.update({rel.nodes[0]['id']: rel['weight']})
        else:
            outgoing_rels.update({rel.nodes[0]['id']: outgoing_rels[rel.nodes[0]['id']]+1})
            outgoing_weights.update({rel.nodes[0]['id']: outgoing_weights[rel.nodes[0]['id']]+rel['weight']})


def balance_score(x, y):
    b_score = ((2 * x * y) / ((x ** 2) + (y ** 2))) * np.log10(min(x, y))
    return b_score


def balance_score_node_filter():
    node_details = {}
    for node_id in list(set(incoming_rels.keys()) | set(outgoing_rels.keys())):
        #logger.info(node_id, incoming_weights.get(node_id, 0), outgoing_weights.get(node_id, 0))
        node_details.update({node_id: balance_score(incoming_weights.get(node_id, 0.1),
                                                    outgoing_weights.get(node_id, 0.1))})
    filtered_node_details = dict(filter(lambda val: val[1] > degree_constant, node_details.items()))
    return filtered_node_details


def create_similarity_matrix(graph, filtered_node_details):
    uv_intersection_query = "MATCH (u {{id: {}}})<-[a:send]-(x)-[b:send]->(v {{id: {}}}) RETURN a,b"
    _uv_intersection_query = "MATCH (u {{id: {}}})-[a:send]->(x)<-[b:send]-(v {{id: {}}}) RETURN a,b"
    uv_single_query = "MATCH (u {{id: {}}})<-[a:send]-() RETURN a"
    _uv_single_query = "MATCH (u {{id: {}}})-[a:send]->() RETURN a"
    w = h = len(filtered_node_details)
    similarity_matrix = [[0 for x in range(w)] for y in range(h)]

    for i in range(len(filtered_node_details)):
        for j in range(len(filtered_node_details)):
            prod_weights_u_n_v = []
            square_weights_u = []
            square_weights_v = []

            _prod_weights_u_n_v = []
            _square_weights_u = []
            _square_weights_v = []
            similarity_matrix[i][j] = 0

            uv_intersection_data = graph.run(uv_intersection_query.format(
                list(filtered_node_details.keys())[i], list(filtered_node_details.keys())[j])).data()
            u_data = graph.run(uv_single_query.format(list(filtered_node_details.keys())[i])).data()
            v_data = graph.run(uv_single_query.format(list(filtered_node_details.keys())[j])).data()

            _uv_intersection_data = graph.run(_uv_intersection_query.format(
                list(filtered_node_details.keys())[i], list(filtered_node_details.keys())[j])).data()
            _u_data = graph.run(_uv_single_query.format(list(filtered_node_details.keys())[i])).data()
            _v_data = graph.run(_uv_single_query.format(list(filtered_node_details.keys())[j])).data()

            # inbound edges
            if uv_intersection_data:
                for data in uv_intersection_data:
                    weights_list = []
                    for val in data.values():
                        weights_list.append(val.get('weight'))
                        prod_weights_u_n_v.append(np.prod(weights_list))

                for data in u_data:
                    square_weights_u.append((data['a'].get('weight'))**2)
                for data in v_data:
                    square_weights_v.append((data['a'].get('weight'))**2)

            # outbound edges
            if _uv_intersection_data:
                for data in _uv_intersection_data:
                    weights_list = []
                    for val in data.values():
                        weights_list.append(val.get('weight'))
                    _prod_weights_u_n_v.append(np.prod(weights_list))
                for data in _u_data:
                    _square_weights_u.append((data['a'].get('weight'))**2)
                for data in _v_data:
                    _square_weights_v.append((data['a'].get('weight'))**2)

                similarity_matrix[i][j] = \
                    (sum(prod_weights_u_n_v) / (math.sqrt(sum(square_weights_u)) * math.sqrt(sum(square_weights_v)))) * \
                    (sum(_prod_weights_u_n_v) / (math.sqrt(sum(_square_weights_u)) * math.sqrt(sum(_square_weights_v))))

                logger.info(f"Nodes {list(filtered_node_details.keys())[i]} - {list(filtered_node_details.keys())[j]}, "
                            f"prodw - {prod_weights_u_n_v}, sqw_u - {square_weights_u}, sqw_v - {square_weights_v}, "
                            f"_prodw - {_prod_weights_u_n_v}, _sqw_u - {_square_weights_u}, "
                            f"_sqw_v - {_square_weights_v}")
    return similarity_matrix


def generate_dense_pairs(filtered_similarity_df):
    dense_pairs = []
    for index in filtered_similarity_df.index:
        for col in filtered_similarity_df.columns:
            if filtered_similarity_df.at[index, col] > 0:
                dense_pairs.append([index, col])

    dense_pairs_set = set(map(frozenset, dense_pairs))
    dense_pairs_list = list(map(list, dense_pairs_set))
    derived_from_dense = []
    _pair = None
    temp = dense_pairs_list.copy()
    for pair in temp:
        unprocessed_pairs = [x for x in temp if x != pair]
        for node in pair:
            for _pair in unprocessed_pairs:
                if node in _pair:
                    derived_from_dense.append(set(pair + _pair))
                    temp.remove(_pair)
    derived_from_dense_set = list(map(set, (set(map(frozenset, derived_from_dense)))))
    return derived_from_dense_set


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    logger.info("...Start Program -- ML Detection using structural similarity...")
    start_time = datetime.datetime.now()
    logger.info(f"Program start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    graph = Graph(password="Genesis2@", uri="bolt://localhost:7687")
    graph.delete_all()
    df = pd.read_csv('transactions.txt', delimiter=',', header=None,
                     names=['trxid', 'sacctid', 'racctid', 'amt', 'time'])
    logger.info(f"Dataframe shape {df.shape}")

    trx_info = namedtuple('trx_info', ['trxid', 'sacctid', 'racctid', 'amt', 'time'])
    amount_threshold = 10000
    allowed_amountdiff = 100
    allowed_timediff = 2
    degree_constant = 0.1
    densepair_constant = 0.2

    logger.info(f"Thresholds - amount_threshold: {amount_threshold}, allowed_amount_difference: {allowed_amountdiff}, "
                f"allowed_time_difference: {allowed_timediff}, degree_constant: {degree_constant}, "
                f"densepair_constant: {densepair_constant}")

    logger.info("Begin matching transactions operation")

    pairs = extract_matching_pairs(df, amount_threshold, allowed_amountdiff, allowed_timediff, trx_info)

    logger.info(f"Total pairs found: {len(pairs)}")
    with open('pairs.txt', 'w') as f:
        for pair in pairs:
            f.write('%s\n' % str(pair))

    create_node_relationships(graph, pairs)

    incoming_rels = {}
    outgoing_rels = {}
    incoming_weights = {}
    outgoing_weights = {}

    calculate_total_weights(graph)

    filtered_node_details = balance_score_node_filter()
    logger.info(f"Filtered node ids with balance score: {filtered_node_details}")
    logger.info(f"Total filtered nodes: {len(filtered_node_details)}")

    similarity_df = pd.DataFrame(create_similarity_matrix(graph, filtered_node_details),
                                 columns=list(filtered_node_details.keys()),
                                 index=list(filtered_node_details.keys()))
    logger.info(f"Similarity matrix dataframe shape: {similarity_df.shape}")
    filtered_similarity_df = similarity_df.loc[:, (similarity_df > densepair_constant).any(axis=1)]
    logger.info(f"Filter similarity matrix with dense_pair_constant of {densepair_constant}. "
                f"New shape: {filtered_similarity_df.shape} ")
    found_ml_groups = generate_dense_pairs(filtered_similarity_df)
    logger.info(f"ML groups/nodes are: {found_ml_groups}")
    logger.info(f"Total ML groups found: {len(found_ml_groups)}")
    end_time = datetime.datetime.now()
    logger.info(f"Program end time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Time elapsed in seconds: {(end_time - start_time).seconds}")
    logger.info("...End Program...")



