
#include <cstdio>
#include <iostream>
#include <map>
#include <algorithm>
#include <cassert>

using namespace std;

#define mp make_pair

map<pair<int,int>, int> my_map;

int main(int argc, char ** argv) {
    int p1, p2;
    double prob;

    int at = 0;

    while (scanf("%d,%d,%lf", &p1, &p2, &prob) != EOF) {
        ++at;
        
        assert (p1 < p2);
        if (!my_map.count(mp(p1, p2))) {
            my_map[mp(p1, p2)] = 1;

            if (prob > 0.0)
                printf("%d,%d,%lf\n", p1, p2, prob);
        }

        if (at && at % 1000000 == 0)
            cerr << at << "\n";
    }

    return 0;
}
