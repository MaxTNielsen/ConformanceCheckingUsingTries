package ee.ut.cs.dsg.confcheck.trie;


import ee.ut.cs.dsg.confcheck.util.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.TreeMap;

public class Trie {

    private final TrieNode root;
    private List<TrieNode> leaves;
    private final HashMap<String, TreeMap<Integer, TrieNode>> warmStart;
    private final int maxChildren;
    private int internalTraceIndex = 0;
    private int size = 0;
    private int numberOfEvents = 0;
    public int minConf = 0;
    public int maxConf = Integer.MIN_VALUE;

    protected HashMap<Integer, String> traceIndexer;

    public Trie(int maxChildren) {
        this.maxChildren = Utils.nextPrime(maxChildren);
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

            if (result == null && current == startFromThisNode)
                return null;
            else if (result == null)
                return current;
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
            if (result == null)
                return null;
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

    public void computeConfidenceCostForAllNodes(String costType) {
        switch (costType) {
            case "standard":
                computeConfidenceCostStandard(this.root);
                break;
            case "avg":
                computeConfidenceCostAVG(this.root);
                break;
            default:
                break;
        }
    }

    private void computeConfidenceCostStandard(TrieNode root_) {
        if (!root_.isEndOfTrace()) {
            for (TrieNode n : root_.getAllChildren()) {
                int confCost = n.getMinPathLengthToEnd();
                if (confCost > maxConf) {
                    maxConf = confCost;
                }
                n.setConfidenceCost(confCost);
                computeConfidenceCostStandard(n);
            }
        }
    }

    private void computeConfidenceCostAVG(TrieNode root_) {
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
                n.setConfidenceCost(confCost);
                computeConfidenceCostAVG(n);
            }
        }
    }

    public void computeScaledConfidenceCost(TrieNode root_) {
        for (TrieNode n : root_.getAllChildren()) {
            if (!n.isEndOfTrace()) {
                double x_std = Math.round((((double) n.getConfidenceCost() - minConf) / (maxConf - minConf)) * 100.0) / 100.0;
                n.setScaledConfCost(x_std);
                computeScaledConfidenceCost(n);
            }
        }
    }

    public List<TrieNode> getLeaves() {
        return leaves;
    }
}