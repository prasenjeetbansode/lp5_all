#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph {
private:
    int numVertices;
    vector<vector<int>> adjList;

public:
    Graph(int numVertices) : numVertices(numVertices) {
        adjList.resize(numVertices);
    }

    void addEdge(int src, int dest) {
        adjList[src].push_back(dest);
    }

    void bfs(int startVertex, int* traversalArray) {
        bool* visited = new bool[numVertices];
        for (int i = 0; i < numVertices; ++i)
            visited[i] = false;

        visited[startVertex] = true;
        queue<int> q;
        q.push(startVertex);

        int index = 0;
        while (!q.empty()) {
            int currentVertex = q.front();
            traversalArray[index++] = currentVertex;
            q.pop();

            #pragma omp parallel for
            for (int i = 0; i < adjList[currentVertex].size(); ++i) {
                int neighbor = adjList[currentVertex][i];
                if (!visited[neighbor]) {
                    visited[neighbor] = true;
                    q.push(neighbor);
                }
            }
        }

        delete[] visited;
    }
};

int main() {
    int numVertices, numEdges;
    cout << "Enter the number of vertices: ";
    cin >> numVertices;
    cout << "Enter the number of edges: ";
    cin >> numEdges;

    Graph graph(numVertices);

    cout << "Enter the edges:\n";
    for (int i = 0; i < numEdges; ++i) {
        int src, dest;
        cin >> src >> dest;
        graph.addEdge(src, dest);
    }

    int startVertex;
    cout << "Enter the starting vertex: ";
    cin >> startVertex;

    int* traversalArray = new int[numVertices];

    double startTime = omp_get_wtime();

    graph.bfs(startVertex, traversalArray);

    double endTime = omp_get_wtime();
    double executionTime = endTime - startTime;

    cout << "BFS traversal: ";
    for (int i = 0; i < numVertices; ++i) {
        cout << traversalArray[i] << " ";
    }
    cout << endl;

    cout << "Execution time: " << executionTime << " seconds" << endl;

    delete[] traversalArray;

    return 0;
}
