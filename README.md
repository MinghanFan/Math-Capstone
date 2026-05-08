# Math 479 Capstone: Random Walk Centrality Project

This project compares classical centrality methods with random-walk centrality methods on two neural networks:

- C. elegans network
- Fruit fly network

The main idea is to see whether random-walk centrality finds different important nodes compared with classical shortest-path centrality.

## What is in this project

We compute and compare:

- classical in-closeness
- classical out-closeness
- classical betweenness
- random-walk in-closeness
- random-walk out-closeness
- random-walk betweenness

We also look at community structure and check whether random walks may be trapped inside communities.

## Folders

- `raw_data/`: original data files
- `data/`: cleaned or processed data
- `elegan_result/`: results for the C. elegans network
- `fly_result/`: results for the fruit fly network
- `community/`: community detection results
- `community_figure_elegan/`: community figures for C. elegans
- `community_figures_fly/`: community figures for fruit fly

## Python files

- `celegans.py`: process C. elegans data
- `fruitfly.py`: process fruit fly data
- `basicstat.py`: compute basic network statistics
- `ccdf.py`: draw degree distribution plots
- `classicstat.py`: compute classical centrality
- `rwclose.py`: compute random-walk closeness by simulation
- `rwbetw.py`: compute random-walk betweenness
- `Compare_method.py`: compare different centrality methods
- `drawing_compare_method.py`: draw comparison plots
- `community.py`: run community detection
- `community_analysis_ele.py`: community analysis for C. elegans
- `community_analysis_fly.py`: community analysis for fruit fly
- `draw_top_in_community.py`: draw top nodes in community figures
