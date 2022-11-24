package ee.ut.cs.dsg.confcheck;

import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.alignment.Move;
import ee.ut.cs.dsg.confcheck.cost.CostFunction;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.trie.TrieNode;

import java.io.Serializable;
import java.util.*;

public abstract class ConformanceChecker implements Serializable {
    protected final Trie modelTrie;
    protected final int logMoveCost ;
    protected final int modelMoveCost ;
    protected PriorityQueue<State> nextChecks;
    protected HashMap<String, State> tracesInBuffer;
    protected HashMap<String, StatesBuffer> statesInBuffer;
    protected int cntr=1;
    protected int maxStatesInQueue;
//    private HashSet<State> seenBefore;
    protected ArrayList<State> states;

    public int getTraceSize() {
        return traceSize;
    }

    protected int traceSize;

    public int getMaxModelTraceSize() {
        return maxModelTraceSize;
    }

    protected int maxModelTraceSize;
    protected int leastCostSoFar  = Integer.MAX_VALUE;

    protected int cleanseFrequency = 100;
    protected int maxTrials=200000;

    protected Trie inspectedLogTraces;
    protected Random rnd;

    public ConformanceChecker(Trie modelTrie, int logCost, int modelCost, int maxCasesInQueue)
    {
        this.modelTrie = modelTrie;
        this.logMoveCost = logCost;
        this.modelMoveCost = modelCost;

        states = new ArrayList<>();
        this.maxStatesInQueue = maxCasesInQueue;
        nextChecks = new PriorityQueue<>(maxCasesInQueue);
        tracesInBuffer = new HashMap<>();
        statesInBuffer = new HashMap<>();
    }

    public abstract Alignment check(List<String> trace);

    protected abstract List<State> handleModelMoves(List<String> traceSuffix, State state, State candidateState);

    protected abstract State handleLogMove(List<String> traceSuffix, State state, String event);

    protected void addStateToTheQueue(State state, State candidateState) {

//        if (seenBefore.contains(state)) {
//            System.out.println("This state has been seen before, skipping it...");
//            return;
//        }
//        else
//            seenBefore.add(state);
//        if (state.getCostSoFar() < 0)
//            return;
        if (cntr==maxStatesInQueue) {
//            System.out.println("Max queue size reached. New state is not added!");
            return;
        }
        cntr++;
        if (nextChecks.size() == maxStatesInQueue)
        {
//            System.out.println("Max queue size reached. New state is not added!");
//           if (state.getCostSoFar() < nextChecks.peek().getCostSoFar())
//            // if (state.getAlignment().getTotalCost() < nextChecks.peek().getAlignment().getTotalCost())
//            {
//                System.out.println(String.format("Adding a good candidate whose cost is %d which is less that the least cost so far %d", state.getAlignment().getTotalCost(), nextChecks.peek().getAlignment().getTotalCost()));
//                System.out.println(String.format("Replacement state suffix length %d, number of model moves %d", state.getTracePostfix().size(), state.getNode().getLevel()));
//                nextChecks.poll();
//                nextChecks.add(state);
//            }
            return;
        }
        if (candidateState != null) {
            if ((state.getAlignment().getTotalCost() + Math.min(Math.abs(state.getTracePostfix().size() - state.getNode().getMinPathLengthToEnd()),Math.abs(state.getTracePostfix().size() - state.getNode().getMaxPathLengthToEnd())))< candidateState.getAlignment().getTotalCost())// && state.getNode().getLevel() > candidateState.getNode().getLevel())
            {

                nextChecks.add(state);
//                states.add(state);
            }
            else {
//                System.out.println(String.format("State is not promising cost %d is greater than the best solution so far %d",(state.getAlignment().getTotalCost()+Math.abs(state.getTracePostfix().size() - state.getNode().getMinPathLengthToEnd())),candidateState.getAlignment().getTotalCost()) );
//                System.out.println("Queue size "+nextChecks.size());
//                System.out.println("Least cost to check next "+nextChecks.peek().getCostSoFar());
            }
        }
        else //if (state.getCostSoFar()< (nextChecks.size() == 0? Integer.MAX_VALUE: nextChecks.peek().getCostSoFar()))
        {
            nextChecks.add(state);
//            states.add(state);
        }
        if (cntr % cleanseFrequency == 0)
        {
            cleanState(candidateState);
            cntr=1;
        }
    }
    private void cleanState(State candidateState)
    {
        int coundDown=cleanseFrequency;
        State current;
        while (nextChecks.size() > cleanseFrequency & coundDown > 0) {
            current = nextChecks.poll();
            if (candidateState != null)
            {
                if ((current.getAlignment().getTotalCost() + Math.abs(current.getTracePostfix().size() - current.getNode().getMinPathLengthToEnd())) >= candidateState.getAlignment().getTotalCost())// && state.getNode().getLevel() > candidateState.getNode().getLevel())
                {
//                    System.out.println(String.format("Removing an old expensive state with cost %d, which is greater than the best solution so far %d",(current.getAlignment().getTotalCost()+Math.abs(current.getTracePostfix().size() - current.getNode().getMinPathLengthToEnd())),candidateState.getAlignment().getTotalCost()) );
//                    System.out.println("Queue size "+nextChecks.size());
                    continue;

                }
            }
            else {
                nextChecks.add(new State(current.getAlignment(), current.getTracePostfix(), current.getNode(), (int) (current.getCostSoFar() + (1 + 10))));
                coundDown--;
            }

        }
        //adjust the frequency of state cleaning

        if(candidateState != null)
        {
            if (leastCostSoFar > candidateState.getAlignment().getTotalCost()) // we couldn't find a better solution since last time, we need to decrease the frequency
            {
                cleanseFrequency = Math.min(100, (cleanseFrequency/10)+100);
            }
            else
                cleanseFrequency *=10;
    //        System.out.println("State cleansing frequency changed to " +cleanseFrequency);
        }

    }

    private int computeCost(int minPathLengthToEnd, int traceSuffixLength, int cumulativeCost, boolean isLogMove)
    {
        int cost = isLogMove? logMoveCost: modelMoveCost;

        // If this is a log move, we have to add 1 to the trie length to end as we have not moved yet from the current node
        // in the trie.
        cost += cumulativeCost + Math.abs( (/*(isLogMove? 1:0) +*/ minPathLengthToEnd) -  traceSuffixLength);
        return cost;
    }

    private int computeCostV2(int minPathLengthToEnd, int traceSuffixLength, int cumulativeCost, boolean isLogMove)
    {
        int cost =0;//isLogMove? logMoveCost: modelMoveCost;


        // If this is a log move, we have to add 1 to the trie length to end as we have not moved yet from the current node
        // in the trie.
//        cost += cumulativeCost + Math.abs( (/*(isLogMove? 1:0) +*/ minPathLengthToEnd) -  traceSuffixLength)+minPathLengthToEnd+traceSuffixLength;
//        cost += Math.max(minPathLengthToEnd+traceSuffixLength - cumulativeCost - Math.abs( (/*(isLogMove? 1:0) +*/ minPathLengthToEnd) -  traceSuffixLength),0);
//

        // Description of the cost: worst case is no alignment at all we have to do a model trace followed by the log trace
        // Then we subtract how far we went into the model model trace which is represented by the misleading name of cumulative cost
        // We also have to subtract how far did we go in the log trace
        cost += maxModelTraceSize+traceSize -(cumulativeCost + (traceSize- traceSuffixLength));//- Math.abs( /*(isLogMove? 1:0) +*/ minPathLengthToEnd -  traceSuffixLength);
        if (cost < 0)
            System.out.println("Cost is negative "+cost +" worst case cost is "+(maxModelTraceSize+traceSize) + "cumulative cost is "+cumulativeCost);
        return cost;

    }

    public Alignment check2(List<String> trace, boolean b, String toString) {
        return null;
    }

    public int tracesInBuffer(){
        return statesInBuffer.size();
    }

    public int statesInBuffer(String id){
        if(statesInBuffer.containsKey(id))
            return statesInBuffer.get(id).getCurrentStates().size();
        return 1;
    }

    public StatesBuffer getTracesInBuffer(String id){
        if(statesInBuffer.containsKey(id))
            return this.statesInBuffer.get(id);
        return new StatesBuffer(new HashMap<>());
    }

    public HashMap<String, StatesBuffer> getCasesInBuffer(){
        return this.statesInBuffer;
    }
}