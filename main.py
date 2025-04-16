import time
from hierarchical_pso import hierarchical_pso

if __name__ == "__main__":
    p = [0.271004, 1.44862, 2.70512, 3.94841, 5.13112, 6.61931, 8.60419, 11.0863, 13.5677, 16.068,
            18.5552, 21.0393, 23.5223, 26.0124, 28.4806, 30.9418, 33.4053, 35.8564, 38.3088, 40.7621,
            42.843, 43.8868, 44.409]
    q = [0.905796, 1.98353, 2.35874, 2.5484, 2.67333, 2.7765, 2.88334, 2.9774, 3.04683, 3.09766,
            3.14708, 3.17517, 3.2241, 3.24116, 3.2549, 3.26587, 3.27946, 3.28687, 3.28825, 3.28971,
            3.30512, 3.30683, 3.30725]

    start_time = time.time()
    result = hierarchical_pso(
        p, q,
        part_n=100000,
        iter_n=1000,
        param_bounds=[[0, 10000000], [0, 10000000]],
        divisions=5,
        w=0.8,
        c1=1.8,
        c2=1.8
    )
    end_time = time.time()

    print(f"Tempo de execução do PSO hierárquico: {end_time - start_time:.5f} segundos")
    print("Resultado final:", result)
