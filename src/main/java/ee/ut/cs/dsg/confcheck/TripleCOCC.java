package ee.ut.cs.dsg.confcheck;

import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.alignment.Move;
import ee.ut.cs.dsg.confcheck.cost.CostFunction;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.trie.TrieNode;
import ee.ut.cs.dsg.confcheck.util.Configuration.MoveType;

import java.util.*;

public class TripleCOCC extends ConformanceChecker {
    final protected int defaultDecayTime = 2;
    final protected int minDecayTime = 1;
    final protected HashMap<String, TreeMap<Integer, TrieNode>> warmStartMap = modelTrie.getWarmStart();
    final protected CostFunction costFunction;
    protected int averageTrieLength;
    protected boolean isStandardAlign;
    protected boolean discountedDecayTime;

    protected float decayTimeMultiplier = 0.3F;

    public TripleCOCC(Trie modelTrie, int logCost, int modelCost, int maxStatesInQueue, boolean discountedDecayTime, boolean isStandardAlign, CostFunction costFunction) {
        super(modelTrie, logCost, modelCost, maxStatesInQueue);
        this.discountedDecayTime = discountedDecayTime;
        this.isStandardAlign = isStandardAlign;
        this.costFunction = costFunction;
        if (discountedDecayTime) {
            averageTrieLength = modelTrie.getAvgTraceLength();
        }
    }

    @Override
    public Alignment check(List<String> trace) {

        return null;
    }

    public HashMap<String, State> check(List<String> trace, String caseId) {
        HashMap<String, State> states = new HashMap<>();
        List<State> syncMoveStates = new ArrayList<>();

        State state;
        State previousState;

        double boundedCost;
        float minCost = isStandardAlign ? 1.0f : 2.0f;

        if (statesInBuffer.containsKey(caseId)) {
            StatesBuffer s = statesInBuffer.get(caseId);
            states = s.getCurrentStates();
            boundedCost = Math.max(getMaxCostOfStates(states), minCost);
        } else {
            boundedCost = minCost;
            states.put(new Alignment().toString(), new State(new Alignment(), new ArrayList<>(), modelTrie.getRoot(), 0.0, computeDecayTime()));
        }

        boundedCost = Math.max(getMaxCostOfStates(states), 2.0f);

        for (String event : trace) {
            for (Map.Entry<String, State> entry : states.entrySet()) {

                previousState = entry.getValue();
                if (previousState.getTracePostfix().size() != 0) {
                    continue; // we are not interested in previous states which already have a suffix (i.e. they already are non-synchronous)
                }

                state = handleSyncMove(event, previousState);
                if (state != null) {
                    // we are only interested in new states which are synced (i.e. they do not have a suffix)
                    syncMoveStates.add(state);
                }
            }

            if (syncMoveStates.size() > 0) {
                for (Map.Entry<String, State> entry : states.entrySet()) {
                    previousState = entry.getValue();
                    if (previousState.getDecayTime() < minDecayTime)
                        states.remove(previousState.getAlignment().toString());
                    else {
                        previousState.setDecayTime(previousState.getDecayTime() - 1);
                        List<String> postfix = new ArrayList<>();
                        postfix.add(event);
                        previousState.addTracePostfix(postfix);
                    }
                }

                for (State syncState : syncMoveStates) {
                    states.put(syncState.getAlignment().toString(), syncState);
                }
                syncMoveStates.clear();

                continue;
            }

            HashMap<String, State> statesToIterate = new HashMap<>(states);
            List<String> traceEvent = new ArrayList<>();
            List<String> traceSuffix;

            for (Map.Entry<String, State> entry : statesToIterate.entrySet()) {
                previousState = entry.getValue();
                int suffixLength = previousState.getTracePostfix().size();

                if (suffixLength == 0) {
                    State logMoveState = handleLogMove(traceEvent, previousState, "");
                }

                traceSuffix = previousState.getTracePostfix();
                traceSuffix.addAll(traceEvent);

                if (suffixLength + 1 + previousState.getWightedSumOfCosts() <= boundedCost) {
                    List<State> modelMoveStates = handleModelMoves(traceSuffix, previousState, null);
                }

                if (previousState.getAlignment().getTraceSize() == 0) {

                }

                if (previousState.getDecayTime() < minDecayTime)
                    states.remove(previousState.getAlignment().toString());
                else {
                    previousState.setDecayTime(previousState.getDecayTime() - 1);
                }
            }
        }
        return null;
    }

    @Override
    protected List<State> handleModelMoves(List<String> traceSuffix, State state, State candidateState) {
        return null;
    }

    @Override
    protected State handleLogMove(List<String> traceEvent, State state, String event) {
        Alignment a = new Alignment(state.getAlignment());
        List<String> suffix = new ArrayList<>(state.getTracePostfix());
        suffix.addAll(traceEvent);

        for (String e : suffix) {
            Move logMove = new Move(e, ">>", 1);
            a.appendMove(logMove);
        }
        return new State(a, new ArrayList<String>(), state.getNode(), updateCostWithConf(state, MoveType.LOG_MOVE, state.getNode(), state.getNode()) + suffix.size(), computeDecayTime());
    }

    protected State handleSyncMove(String event, State state) {
        TrieNode fromNode = state.getNode();
        TrieNode toNode = fromNode.getChild(event);

        Alignment a = new Alignment(state.getAlignment());
        Move syncMove;

        if (toNode == null) {
            return null;
        } else {
            syncMove = new Move(event, event, 0);
            a.appendMove(syncMove);
            return new State(a, new ArrayList<>(), toNode, updateCostWithConf(state, MoveType.SYNCHRONOUS_MOVE, fromNode, toNode),
                    state, computeDecayTime());
        }
    }

    protected List<State> handleWarmStartMove(String event, State state, float boundedCost) {
        TreeMap<Integer, TrieNode> warmStartNodes = warmStartMap.get(event);
        List<State> warmStartStates = new ArrayList<>();
        List<String> suffix = new ArrayList<>();

        for (Map.Entry<Integer, TrieNode> entry : warmStartNodes.entrySet()) {
            int completenessCost = entry.getKey();
            TrieNode warmStartNode = entry.getValue();

            if (completenessCost <= boundedCost) {
                Alignment a = new Alignment();
                a.appendMove(new Move(warmStartNode.getContent(), warmStartNode.getContent(), 0));

                if (state.getTracePostfix().size() == 0 & warmStartNode.getChild(event) != null) {
                    a.appendMove(new Move(warmStartNode.getContent(), warmStartNode.getContent(), 0));
                    a.appendMove((new Move(event, event, 0)));
                    warmStartStates.add(new State(a, new ArrayList<>(), warmStartNode.getChild(event), (double) completenessCost, computeDecayTime()));

                } else if (completenessCost + state.getTracePostfix().size() + 1 <= boundedCost) {
                    a.appendMove(new Move(warmStartNode.getContent(), warmStartNode.getContent(), 0));
                    State s = new State(a, state.getTracePostfix(), warmStartNode,
                            (double) completenessCost + warmStartNode.getScaledConfCost(), computeDecayTime());
                    State logMove = handleLogMove(new ArrayList<>(), s, "");
                    warmStartStates.add(logMove);

                }
            }
        }
        return warmStartStates;
    }

    protected double getMaxCostOfStates(HashMap<String, State> s) {
        double maxCost = Float.MAX_VALUE;
        Collection<State> states = s.values();
        for (State st : states) {
            if (st.getWightedSumOfCosts() <= maxCost)
                maxCost = st.getWightedSumOfCosts();
        }
        return maxCost;
    }

    protected double updateCostWithConf(State state, MoveType mv, TrieNode prevEta, TrieNode eta) {
        if (mv == MoveType.LOG_MOVE) {
            return state.getWightedSumOfCosts();
        }
        return state.getWightedSumOfCosts() - prevEta.getScaledConfCost() + eta.getScaledConfCost();
    }

    protected int computeDecayTime() {
        if (discountedDecayTime)
            return Math.max(Math.round(averageTrieLength * decayTimeMultiplier), defaultDecayTime);
        return defaultDecayTime;
    }
}
