import pandas as pd
from hierarchical_clustering import HierarchicalClustering

def main():

    data = pd.read_csv("test_data.csv")
    hc = HierarchicalClustering(k=2)
    cluster_assignments = hc.fit_predict(data.to_numpy())
    print(cluster_assignments)


if __name__ == "__main__":
    main()
