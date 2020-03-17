# ffe-spark
Fire Following Earthquake (FFE) experiments with Apache Spark

The aim of the project is to explore the application of Spark and GeoMesa to this problem domain. 

There may be two key applications of interest:

## 1) Finding adjacent vertices

The naive algorithm for finding adjacent vertices (the flammable objects) by distance has O(n2) cost, making it impractical for medim-large scale madelling. Can we do better e.g. O(nlogn) using geo-indexed algorithms e.g. those in GeoMesa for Spark.

ref:

 - https://blog.mapbox.com/a-dive-into-spatial-search-algorithms-ebd0c5e39d2a
 -  Fast Algorithm for Finding Maximum Distance with
Space Subdivision in E2 https://arxiv.org/pdf/1708.02758.pdf
 - https://www.geeksforgeeks.org/closest-pair-of-points-using-divide-and-conquer-algorithm/
 - https://www.geomesa.org/assets/outreach/SpatioTemporalIndexing_IEEEcopyright.pdf
 - sa.org/documentation/user/spark/spark_jts.html demonstrates use of geomesa in Spark using dataframes & CSV only.

## 2) Modelling

Modelling the spread of fire over time and eventual fuel exhaustion could be well suited to a Graph data model using message passing. These are available in Spark GraphX so here we can compare this to the current approach using for loops and a Pandas dataframe. 

 - https://spark.apache.org/docs/latest/graphx-programming-guide.html#graph-builders
 - https://neo4j.com/developer/apache-spark/
 - https://spark.apache.org/docs/latest/graphx-programming-guide.html#pregel-api 
   "At a high level the Pregel operator in GraphX is a bulk-synchronous parallel messaging abstraction constrained to the topology of the graph. The Pregel operator executes in a series of super steps in which vertices receive the sum of their inbound messages from the previous super step,..."  
