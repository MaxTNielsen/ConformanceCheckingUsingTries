package ee.ut.cs.dsg.confcheck.trie;


import ee.ut.cs.dsg.confcheck.util.PredictionsClient;
import ee.ut.cs.dsg.confcheck.util.AlphabetService;
import ee.ut.cs.dsg.confcheck.util.Utils;

import java.util.*;
import java.util.stream.Collectors;

public class Trie {

    private final TrieNode root;
    private List<TrieNode> leaves;
    private final HashMap<String, TreeMap<Integer, TrieNode>> warmStart;
    private final int maxChildren;
    private int internalTraceIndex = 0;
    private int size = 0;
    private int numberOfEvents = 0;
    private boolean isWeighted;
    public int minConf = 0;
    public double maxConf = Integer.MIN_VALUE;

    Map<String, Double> prefixProbCache = new HashMap<>();

    protected HashMap<Integer, String> traceIndexer;

    private PredictionsClient p;


    private AlphabetService service;

    public Trie(int maxChildren, AlphabetService service) {
        this.maxChildren = Utils.nextPrime(maxChildren);
        this.service = service;
        root = new TrieNode("dummy", maxChildren, Integer.MAX_VALUE, Integer.MIN_VALUE, false, null);
        traceIndexer = new HashMap<>();
        leaves = new ArrayList<>();
        // initial capacity for hashmap = maxChildren
        warmStart = new HashMap<>(maxChildren);
    }

    public int getMaxChildren() {
        return maxChildren;
    }

    public void addTrace(List<String> trace) {
        ++internalTraceIndex;
        addTrace(trace, internalTraceIndex);

    }

    public AlphabetService getService() {
        return service;
    }

    public void addTrace(List<String> trace, int traceIndex) {
        TrieNode current = root;
        int minLengthToEnd = trace.size();
        if (minLengthToEnd > 0) {
            StringBuilder sb = new StringBuilder(trace.size());
            for (String event : trace) {

                /*if(event.equals("O"))
                    System.out.println("");*/

                current.addLinkedTraceIndex(traceIndex);
                TrieNode child = new TrieNode(event, maxChildren, minLengthToEnd - 1, minLengthToEnd - 1, minLengthToEnd - 1 == 0 ? true : false, current);
                TrieNode returned;
                returned = current.addChild(child);
                if (returned == child) // we added a new node to the trie
                {
                    size++;
                }
                current = returned;

                minLengthToEnd--;
                sb.append(event);
                if (returned.isEndOfTrace()) {
                    leaves.add(returned);
                } else {
                    // build warm start map
                    if (warmStart.containsKey(current.getContent())) {
                        warmStart.get(current.getContent()).put(current.getLevel() - 1, current);
                    } else {
                        warmStart.put(current.getContent(), new TreeMap<>());
                        warmStart.get(current.getContent()).put(current.getLevel() - 1, current);
                    }
                }
            }
            current.addLinkedTraceIndex(traceIndex);
            numberOfEvents += sb.length();
            traceIndexer.put(traceIndex, sb.toString());
        }

    }

    public TrieNode getRoot() {
        return root;
    }

    public String toString() {
        return root.toString();
    }

    public String getTrace(int index) {
        return traceIndexer.get(index);
    }

    public void printTraces() {
//        StringBuilder result = new StringBuilder();
//        TrieNode current;
//        for (TrieNode  leaf: leaves)
//        {
//            current = leaf;
//            result = new StringBuilder();
//            while (current != root)
//            {
//                result.append(current.getContent()+",");
//                current = current.getParent();
//            }
//
//            System.out.println(result.reverse().toString());
//        }
        for (String s : traceIndexer.values())
            System.out.println(s);
    }

    /**
     * This method finds the deepest node in the trie that provides the longest prefix match to the trace.
     * If there is no match at all, the method returns null.
     *
     * @param trace is a list of strings that define the trace to search a match for
     * @return a trie node
     */
    public TrieNode match(List<String> trace, TrieNode startFromThisNode) {
        TrieNode current = startFromThisNode;
        TrieNode result;
        int size = trace.size();
        int lengthDifference = Integer.MAX_VALUE;
        for (int i = 0; i < size; i++)
//        for (String event : trace)
        {
            result = current.getChild(trace.get(i));
            // result = current.getChildWithLeastPathLengthDifference(trace.get(i), size - i);

            if (result == null && current == startFromThisNode) return null;
            else if (result == null) return current;
            else {
                TrieNode result2 = result;
                //result2 = result.getChildWithLeastPathLengthDifference(size-(i+1));

//                if (Math.abs(result2.getMinPathLengthToEnd() - (size - (i+1))) <= lengthDifference)
                //               {
                // we still have a promising direction
                current = result;
//                    lengthDifference = Math.abs(result.getMinPathLengthToEnd() - (size - (i+1)));
//                }
//                else
//                    return current.getParent();
            }
        }
        return current;
    }

    public TrieNode match(List<String> trace) {
        return match(trace, root);
    }

    public TrieNode matchCompletely(List<String> trace, TrieNode startFromThisNode) {
        TrieNode current = startFromThisNode;
        TrieNode result;
        int size = trace.size();
        for (int i = 0; i < size; i++) {
            result = current.getChild(trace.get(i));
            if (result == null) return null;
            else {
                TrieNode result2 = result;
                current = result;
            }
        }
        return current;
    }

    public int getMaxTraceLength() {
        int maxLength = leaves.stream().map(node -> node.getLevel()).reduce(Integer.MIN_VALUE, (minSoFar, element) -> Math.max(minSoFar, element));
        return maxLength;
    }

    public int getMinTraceLength() {
        int minLength = leaves.stream().map(node -> node.getLevel()).reduce(Integer.MAX_VALUE, (minSoFar, element) -> Math.min(minSoFar, element));
        return minLength;
    }

    public int getAvgTraceLength() {
        int sumlength = leaves.stream().map(node -> node.getLevel()).reduce(0, (subtotal, element) -> subtotal + element);
        return sumlength / leaves.size();
    }

    public int getSize() {
        return size;
    }

    public int getNumberOfEvents() {
        return numberOfEvents;
    }

    public HashMap<String, TreeMap<Integer, TrieNode>> getWarmStart() {
        return warmStart;
    }

    public TrieNode getNodeOnShortestTrace() {
        int currentMinLevel = 99999;
        TrieNode currentMinNode = null;
        for (TrieNode n : leaves) {
            if (n.getLevel() < currentMinLevel) {
                currentMinNode = n;
                currentMinLevel = n.getLevel();
            }
        }
        return currentMinNode;
    }

    public List<TrieNode> getLeavesFromNode(TrieNode startNode, int maxLevel) {
        List<TrieNode> result = new ArrayList<>();
        TrieNode currentNode;
        int startNodeLevel = startNode.getLevel();
        int currentNodeLevel;
        for (TrieNode n : leaves) {

            currentNodeLevel = n.getLevel();
            currentNode = n;
            if (currentNodeLevel > startNodeLevel & n.getLevel() <= maxLevel) {
                if (result.contains(n)) {
                    continue;
                }
                while (currentNodeLevel > startNodeLevel) {
                    currentNode = currentNode.getParent();
                    currentNodeLevel = currentNode.getLevel();
                    if (currentNodeLevel == startNodeLevel & currentNode == startNode) {
                        result.add(n);
                    }
                }
            }
        }
        return result;
    }

    public void computeConfidenceCostForAllNodes(String costType, HashMap<String, String> urls, String logName) {
        isWeighted = urls.size() > 0;

        if (isWeighted) {
            this.p = new PredictionsClient(urls);
            Map<String, String> params = new HashMap<>();
            params.put("filename", logName);
            if (p.initModel("init", params) != 200) {
                costType = "";
                System.out.println("Model not initialized and confidence cost skipped all together");
            }
        }

        switch (costType) {
            case "standard":
                computeConfidenceCostStandard(this.root);
                break;
            case "avg":
                //computeConfidenceCostAVG(this.root, isWeighted);
                computeAndUsePrefProbOnly(this.root);
                //computeScaledConfidenceCost(this.root, false);
                if (isWeighted) {
                    prefixProbCache = null;
                    System.gc();
                }
                break;
            default:
                break;
        }

        if (isWeighted) {
            p.closeConnection();
        }
    }

    private void computeConfidenceCostStandard(TrieNode root_) {
        if (!root_.isEndOfTrace()) {
            for (TrieNode n : root_.getAllChildren()) {
                int confCost = n.getMinPathLengthToEnd();
                if (confCost > maxConf) {
                    maxConf = confCost;
                }
                n.setScaledConfCost(confCost);
                computeConfidenceCostStandard(n);
            }
        }
    }

    private void computeConfidenceCostAVG(TrieNode root_, boolean isWeighted) {
        if (!root_.isEndOfTrace()) {
            for (TrieNode n : root_.getAllChildren()) {
                List<TrieNode> leavesOfChildNode = getLeavesFromNode(n, Integer.MAX_VALUE);
                if (leavesOfChildNode.size() == 0) {
                    continue;
                }
                int confCost = leavesOfChildNode.stream().map(x -> x.getLevel() - n.getLevel()).mapToInt(Integer::intValue).sum() / leavesOfChildNode.size();
                if (confCost > maxConf) {
                    maxConf = confCost;
                }
                if (isWeighted) {
                    double weightedConfCost = confCost * getPrefProb(n);
                    n.setConfidenceCost(weightedConfCost);
                    computeConfidenceCostAVG(n, true);
                } else {
                    n.setConfidenceCost(confCost);
                    computeConfidenceCostAVG(n, false);
                }
            }
        }
    }

    public void computeScaledConfidenceCost(TrieNode root_, boolean isWeighted) {
        for (TrieNode n : root_.getAllChildren()) {
            if (!n.isEndOfTrace()) {
                double x_std = Math.round((((double) n.getConfidenceCost() - minConf) / (maxConf - minConf)) * 100.0) / 100.0;
                if (isWeighted) {
                    double weightedConfCost = x_std * getPrefProb(n);
                    n.setScaledConfCost(weightedConfCost);
                    computeScaledConfidenceCost(n, true);
                } else {
                    n.setScaledConfCost(x_std);
                    computeScaledConfidenceCost(n, false);
                }
            }
        }
    }

    public void computeAndUsePrefProbOnly(TrieNode root_) {
        for (TrieNode n : root_.getAllChildren()) {
            if (!n.isEndOfTrace()) {
                double prefProb = getPrefProb(n);
                if (prefProb > maxConf) {
                    maxConf = prefProb;
                }
                n.setScaledConfCost(getPrefProb(n));
                computeAndUsePrefProbOnly(n);
            }
        }
    }

    private double getPrefProb(TrieNode target) {
        String[] prefixNodes = target.getPrefix().split("->");
        String prefKey = Arrays.toString(prefixNodes);
        if (prefixProbCache.containsKey(prefKey)) {
            return prefixProbCache.get(prefKey);
        } else {
            StringBuilder jsonString = new StringBuilder();
            jsonString.append("{\"trace\":[");
            if (prefixNodes.length > 1) {
                int i = 0;
                for (; i < prefixNodes.length - 2; i++) {
                    jsonString.append("\"").append(service.deAlphabetize(prefixNodes[i].toCharArray()[0])).append("\",");
                }
                jsonString.append("\"").append(service.deAlphabetize(prefixNodes[i].toCharArray()[0])).append("\"");
            } else {
                jsonString.append("\"").append(service.deAlphabetize(target.getContent().toCharArray()[0])).append("\"");
            }
            jsonString.append("], \"target\":\"").append(service.deAlphabetize(target.getContent().toCharArray()[0])).append("\"}");
            double prefProb = Math.round((1 - p.getPrefixProb("pred", jsonString.toString())) * 1000.0) / 1000.0;
            prefixProbCache.put(prefKey, prefProb);
            return prefProb;
        }
    }
}
