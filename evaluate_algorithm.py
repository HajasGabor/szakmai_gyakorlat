import numpy as np
import pandas as pd
from scipy.spatial import distance
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray

# Ground truth középpontok beolvasása a CSV fájlból
def load_csv_centroids(csv_file):
    data = pd.read_csv(csv_file, delimiter=';')  # Alapértelmezett elválasztó a ';'
    centroids = data[['meanX', 'meanY', 'meanZ']].values  # Középpontok koordinátáinak kivétele
    
    # Z értékek beállítása 0-ra
    centroids[:, 2] = 0  # A meanZ (z) értékek mindegyikét 0-ra állítja
    
    return centroids

# Legközelebbi szomszéd párosítás (ground truth középpontok és algoritmus pontok)
def nearest_neighbor_matching(ground_truth, algo_centroids, threshold):
    matches = []
    exact_matches = 0
    for gt_point in ground_truth:
        distances = distance.cdist([gt_point], algo_centroids, 'euclidean')
        closest_idx = np.argmin(distances)
        closest_dist = distances[0][closest_idx]
        
        # Ellenőrizzük, hogy a legközelebbi pont távolsága kisebb-e, mint a küszöbérték
        if closest_dist <= threshold:
            exact_matches += 1
        
        matches.append((gt_point, algo_centroids[closest_idx], closest_dist))  # Távolság is
    return matches, exact_matches

# Eredmény kiértékelése
def evaluate_matches(matches, exact_matches, total_algo_points):
    total_distance = 0
    for gt_point, algo_point, dist in matches:
        total_distance += dist
    mean_error = total_distance / len(matches)
    
    # Százalékos találati arány számítása
    accuracy_percentage = (exact_matches / total_algo_points) * 100
    return mean_error, accuracy_percentage

# Node osztály a ROS 2-ben a MarkerArray topic feliratkozásához
class MarkerArraySubscriber(Node):
    def __init__(self, ground_truth_centroids, distance_threshold):
        super().__init__('marker_array_subscriber')
        self.subscription = self.create_subscription(
            MarkerArray,
            '/clustered_marker',  # A topic neve, amelyre feliratkozik
            self.listener_callback,
            10
        )
        self.ground_truth_centroids = ground_truth_centroids
        self.distance_threshold = distance_threshold

    def listener_callback(self, msg):
        # Az algoritmus által kiszámított középpontok beolvasása a MarkerArray üzenetből
        algo_centroids = []
        for marker in msg.markers:
            if marker.ns == 'cluster_center':  # Filter only 'cluster_center' markers
                algo_centroids.append([marker.pose.position.x, marker.pose.position.y, 0])  # Z érték 0-ra állítása
        algo_centroids = np.array(algo_centroids)

        # Legközelebbi szomszéd párosítás
        matches, exact_matches = nearest_neighbor_matching(self.ground_truth_centroids, algo_centroids, self.distance_threshold)

        # Kiértékelés
        mean_error, accuracy_percentage = evaluate_matches(matches, exact_matches, len(algo_centroids))

        # Eredmények kiírása
        self.get_logger().info(f"Átlagos párosítási hiba: {mean_error}")
        self.get_logger().info(f"Pontos egyezés találati arány: {accuracy_percentage}%")
        self.get_logger().info(f"Pontos egyezések száma: {exact_matches}")

        # Leállítás az első kiértékelés után
        rclpy.shutdown()

def main(args=None):
    # Ground truth betöltése a CSV fájlból
    csv_file = "ground_truth_points.csv"
    ground_truth_centroids = load_csv_centroids(csv_file)
    distance_threshold = 0.5  # Távolságküszöb, amit pontos egyezésnek tekintünk

    # ROS 2 inicializálás
    rclpy.init(args=args)

    # Node inicializálása és futtatása
    marker_array_subscriber = MarkerArraySubscriber(ground_truth_centroids, distance_threshold)
    rclpy.spin(marker_array_subscriber)

    # Node és ROS 2 leállítása
    marker_array_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
