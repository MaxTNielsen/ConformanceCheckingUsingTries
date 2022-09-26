package ee.ut.cs.dsg.confcheck;

import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.alignment.Move;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.trie.TrieNode;
import ee.ut.cs.dsg.confcheck.util.Configuration.MoveType;

import java.util.*;

public class TripleCOCC extends ConformanceChecker {
    final protected int defaultDecayTime = 2;
    final protected int minDecayTime = 1;
    final protected HashMap<String, TreeMap<Integer, TrieNode>> warmStartMap = modelTrie.getWarmStart();
    protected int averageTrieLength;
    protected boolean isStandardAlign;
    protected boolean discountedDecayTime;
    protected boolean replayWithLogMoves = true;
    protected float decayTimeMultiplier = 0.3F;

    public TripleCOCC(Trie modelTrie, int logCost, int modelCost, int maxStatesInQueue, boolean discountedDecayTime, boolean isStandardAlign) {
        super(modelTrie, logCost, modelCost, maxStatesInQueue);
        this.discountedDecayTime = discountedDecayTime;
        this.isStandardAlign = isStandardAlign;

        if (!isStandardAlign) {
            modelTrie.computeConfidenceCostForAllNodes("avg");
            modelTrie.computeScaledConfidenceCost(modelTrie.getRoot());
        }

        if (discountedDecayTime) {
            averageTrieLength = modelTrie.getAvgTraceLength();
        }
    }

    @Override
    public Alignment check(List<String> trace) {
        return null;
    }

    @Override
    protected List<State> handleModelMoves(List<String> traceSuffix, State state, State candidateState) {
        return null;
    }

    public HashMap<String, State> check(List<String> trace, String caseId) {
        HashMap<String, State> states = new HashMap<>();
        List<State> syncMoveStates = new ArrayList<>();

        StatesBuffer sBuffer = null;

        State state;
        State previousState;

        double boundedCost;
        double minCost = isStandardAlign ? 1.0 : 2.0;

        if (statesInBuffer.containsKey(caseId)) {
            sBuffer = statesInBuffer.get(caseId);
            states = sBuffer.getCurrentStates();
            boundedCost = Math.max(getMaxCostOfStates(states) + 1, minCost);
        } else {
            boundedCost = minCost;
            states.put(new Alignment().toString(), new State(new Alignment(), new ArrayList<>(), modelTrie.getRoot(), 0.0, computeDecayTime()));
        }

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
                Iterator<Map.Entry<String, State>> iter = states.entrySet().iterator();
                while (iter.hasNext()) {
                    previousState = iter.next().getValue();
                    if (previousState.getDecayTime() < minDecayTime)
                        iter.remove();
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
            traceEvent.add(event);

            List<String> traceSuffix;
            List<State> nonSynchronousStates = new ArrayList<>();

            for (Map.Entry<String, State> entry : statesToIterate.entrySet()) {
                previousState = entry.getValue();
                int suffixLength = previousState.getTracePostfix().size();

                if (suffixLength == 0) {
                    State logMoveState = handleLogMove(traceEvent, previousState, "");
                    states.put(logMoveState.getAlignment().toString(), logMoveState);
                }

                traceSuffix = previousState.getTracePostfix();
                traceSuffix.addAll(traceEvent);

                if (suffixLength + previousState.getWightedSumOfCosts() <= boundedCost) {
                    nonSynchronousStates.addAll(handleModelMoves(traceSuffix, previousState, null, boundedCost));
                }

                if (previousState.getAlignment().getTraceSize() == 0) {
                    nonSynchronousStates.addAll(handleWarmStartMove(traceEvent, previousState, boundedCost));
                }

                if (previousState.getDecayTime() < minDecayTime)
                    states.remove(previousState.getAlignment().toString());
                else {
                    previousState.setDecayTime(previousState.getDecayTime() - 1);
                }
            }
            for (State s : nonSynchronousStates) {
                states.put(s.getAlignment().toString(), s);
            }
        }

        if (sBuffer == null) {
            sBuffer = new StatesBuffer(states);
        } else {
            sBuffer.setCurrentStates(states);
        }

        statesInBuffer.put(caseId, sBuffer);
        return states;
    }

    protected List<State> handleModelMoves(List<String> traceSuffix, State state, State candidateState, double boundedCost) {
        TrieNode matchNode;
        Alignment alg;
        //make a new list and add to it
        List<String> suffixToCheck = new ArrayList<>(traceSuffix);
        int lookAheadLimit = traceSuffix.size();
        List<TrieNode> currentNodes = new ArrayList<>();
        List<TrieNode> childNodes = new ArrayList<>();
        List<TrieNode> matchingNodes = new ArrayList<>();
        List<State> matchingStates = new ArrayList<>();
        currentNodes.add(state.getNode());

        while (lookAheadLimit > 0) {
            // from current level, fetch all child nodes
            for (TrieNode n : currentNodes) {
                childNodes.addAll(n.getAllChildren());
            }
            // for all child nodes, try to get a substring match
            for (TrieNode n : childNodes) {
                matchNode = modelTrie.matchCompletely(suffixToCheck, n);
                if (matchNode != null) {
                    matchingNodes.add(matchNode);
                }
            }
            // something has matched, we will not look further
            if (matchingNodes.size() > 0) {
                break;
            }
            // no match, so child nodes become current nodes, and we reduce look ahead
            currentNodes.clear();
            currentNodes.addAll(childNodes);
            childNodes.clear();
            lookAheadLimit--;

            //if lookAhead is exhausted, but we can split suffix
            if (lookAheadLimit == 0 & suffixToCheck.size() > 1 & replayWithLogMoves) {
                suffixToCheck.remove(0);
                lookAheadLimit = suffixToCheck.size();
                currentNodes.clear();
                currentNodes.add(state.getNode());
            }
        }

        if (matchingNodes.size() == 0) {
            //we didn't find any match, return empty array
        } else {
            // iterate back from matchingNode until parent = state.getNode
            // because we need correct alignment and cost
            for (TrieNode n : matchingNodes) {
                alg = state.getAlignment();
                double cost = state.getWightedSumOfCosts();
                TrieNode currentNode = n;
                TrieNode parentNode = n.getParent();
                TrieNode lastMatchingNode = state.getNode();
                List<Move> moves = new ArrayList<>();
                boolean makeLogMoves = false;

                // first find all sync moves, then add model moves (parent does not match event), then add log moves (events still remaining in traceSuffix)
                for (int i = traceSuffix.size(); --i >= 0; ) {
                    String event = traceSuffix.get(i);
                    if (event.equals(currentNode.getContent())) {
                        Move syncMove = new Move(event, event, 0);
                        moves.add(0, syncMove);
                        currentNode = parentNode;
                        parentNode = currentNode.getParent();
                        if (i > 0) {
                            continue; // there could still be more sync moves
                        }
                    } else {
                        makeLogMoves = true;
                    }
                    // we either have a non-sync move or we have exhausted the suffix.
                    // so we need to add model moves (and log moves if applicable)

                    // we first iterate until we get to the lastMatchingNode
                    while (!currentNode.equals(lastMatchingNode)) {
                        Move modelMove = new Move(">>", currentNode.getContent(), 1);
                        cost++;
                        moves.add(0, modelMove);
                        currentNode = parentNode;
                        if (currentNode.getLevel() == 0) { //we have reached the root node
                            break;
                        }
                        parentNode = currentNode.getParent();
                    }
                    // we also add all log moves now
                    while (makeLogMoves & i >= 0) {
                        event = traceSuffix.get(i);
                        Move logMove = new Move(event, ">>", 1);
                        cost++;
                        moves.add(0, logMove);
                        i--;
                    }
                }
                // matching states
                if (n.getScaledConfCost() + cost <= boundedCost) {
                    for (Move m : moves) {
                        alg.appendMove(m);
                    }
                    matchingStates.add(new State(alg, new ArrayList<>(), n, n.getScaledConfCost() + cost, computeDecayTime()));
                }
            }
        }
        return matchingStates;
    }

    @Override
    protected State handleLogMove(List<String> traceEvent, State state, String event) {
        Alignment a = new Alignment(state.getAlignment());
        List<String> suffix = new ArrayList<>(state.getTracePostfix());
        suffix.addAll(traceEvent);

        for(String e : suffix) {
            Move logMove = new Move(e, ">>", 1);
            a.appendMove(logMove);
        }

        return new State(a, new ArrayList<>(), state.getNode(), updateCost(state.getWightedSumOfCosts(), MoveType.LOG_MOVE, state.getNode(), state.getNode()) + suffix.size(), computeDecayTime());
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
            return new State(a, new ArrayList<>(), toNode, updateCost(state.getWightedSumOfCosts(), MoveType.SYNCHRONOUS_MOVE, fromNode, toNode),
                    state, computeDecayTime());
        }
    }

    protected List<State> handleWarmStartMove(List<String> event, State state, double boundedCost) {
        List<State> warmStartStates = new ArrayList<>();
        List<String> suffix = state.getTracePostfix();
        suffix.addAll(event);
        TreeMap<Integer, TrieNode> warmStartNodes = warmStartMap.get(suffix.get(0));
        System.out.println("in warmstart");
        System.out.printf("%d", boundedCost);
        for (Map.Entry<Integer, TrieNode> entry : warmStartNodes.entrySet()) {
            int completenessCost = entry.getKey();
            TrieNode warmStartNode = entry.getValue();

            // we only consider warm-start moves that are within the bounded cost
            if (completenessCost + warmStartNode.getScaledConfCost() <= boundedCost) {
                Alignment a = new Alignment();
                a.appendMove(new Move(warmStartNode.getContent(), warmStartNode.getContent(), 0));

                // we can make a synchronous move on the warm-start node if the suffix is empty
                if (state.getTracePostfix().size() == 0) {
                    warmStartStates.add(new State(a, new ArrayList<>(), warmStartNode, updateCost(completenessCost + warmStartNode.getScaledConfCost(), MoveType.SYNCHRONOUS_MOVE, warmStartNode, warmStartNode), computeDecayTime()));

                // again, we check if the maximum cost will be exceeded by computing warm-start log or model prefix-alignments
                } else if (completenessCost + warmStartNode.getScaledConfCost() + suffix.size() - 1 <= boundedCost) {
                    System.out.println(suffix.remove(0).toString());
                    System.out.println(warmStartNode.toString());

                    State s = new State(a, suffix, warmStartNode,
                            updateCost(completenessCost + warmStartNode.getScaledConfCost(), MoveType.LOG_MOVE, warmStartNode, warmStartNode), computeDecayTime());

                    // compute log move state from warm-start node
                    State logMove = handleLogMove(new ArrayList<>(), s, "");
                    warmStartStates.add(logMove);

                    // compute model move state from warm-start node
                    warmStartStates.addAll(handleModelMoves(suffix, s, null, boundedCost));
                }
            }
        }
        return warmStartStates;
    }

    public State getCurrentOptimalState(String caseId, boolean finalState) {
        StatesBuffer caseStatesInBuffer;
        HashMap<String, State> currentStates;

        double  currentCost;
        int decayTime;
        State newestState = null;
        State oldestState = null;
        List<State> newestStates = new ArrayList<>();
        List<State> oldestStates = new ArrayList<>();

        if (statesInBuffer.containsKey(caseId)) {
            caseStatesInBuffer = statesInBuffer.get(caseId);
            currentStates = caseStatesInBuffer.getCurrentStates();
            List<State> statesList = new ArrayList<>(currentStates.values());
            for (State s : statesList) {
                if (finalState) {

                    // if it is end of trace and end of model, then return that state immediately
                    if (
                            ((s.getTracePostfix().size() + s.getNode().getMinPathLengthToEnd()) == 0 ||
                                    (s.getTracePostfix().size() == 0 && s.getNode().isEndOfTrace()))
                                    && s.getWightedSumOfCosts() == 0
                    ) {
                        //System.out.printf("End of trace %n state:%s", s.toString());
                        return s;
                    }

                    // we are interested in the oldest and newest states
                    if (newestState == null
                            || (s.getDecayTime() > newestState.getDecayTime() & s.getTracePostfix().size() < newestState.getTracePostfix().size())
                            || (s.getDecayTime() > newestState.getDecayTime() & s.getTracePostfix().size() == newestState.getTracePostfix().size())
                            || (s.getDecayTime() == newestState.getDecayTime() & s.getTracePostfix().size() < newestState.getTracePostfix().size())
                    ) {
                        newestState = s;
                        newestStates.clear();
                        newestStates.add(s);
                    } else if ((s.getDecayTime() == newestState.getDecayTime() & s.getTracePostfix().size() == newestState.getTracePostfix().size())) {
                        newestStates.add(s);
                    }

                    if (oldestState == null
                            || (s.getDecayTime() < oldestState.getDecayTime() & s.getTracePostfix().size() > oldestState.getTracePostfix().size())
                            || (s.getDecayTime() < oldestState.getDecayTime() & s.getTracePostfix().size() == oldestState.getTracePostfix().size())
                            || (s.getDecayTime() == oldestState.getDecayTime() & s.getTracePostfix().size() > oldestState.getTracePostfix().size())
                    ) {
                        oldestState = s;
                        oldestStates.clear();
                        oldestStates.add(s);
                    } else if ((s.getDecayTime() == oldestState.getDecayTime() & s.getTracePostfix().size() == oldestState.getTracePostfix().size())) {
                        oldestStates.add(s);
                    }

                } else {
                    // just want to return the latest / current state. This state is prefix-alignment type, not full alignment
                    if (discountedDecayTime) {
                        decayTime = Math.max(Math.round((averageTrieLength - s.getAlignment().getTraceSize()) * decayTimeMultiplier), minDecayTime);
                    } else {
                        decayTime = minDecayTime;
                    }
                    if (s.getDecayTime() == decayTime & s.getTracePostfix().size() == 0) {
                        return s;
                    }
                }

            }

            // calculate cost from newestState
            double optimalCost = 9999999;
            Alignment optimalAlg = null;
            TrieNode optimalNode = null;

            for (State s : newestStates) {
                currentCost = s.getWightedSumOfCosts();

                Alignment alg = s.getAlignment();
                TrieNode currentNode = s.getNode();
                List<String> postfix = new ArrayList<>(s.getTracePostfix());
                // add log moves - should be none
                while (postfix.size() > 0) {
                    Move m = new Move(postfix.get(0), ">>", 1);
                    alg.appendMove(m, 1);
                    currentCost++;
                    postfix.remove(0);
                }

                // add model moves
                if (!currentNode.isEndOfTrace()) {
                    while (currentNode.getMinPathLengthToEnd() > 0) {
                        currentNode = currentNode.getChildOnShortestPathToTheEnd();
                        Move m = new Move(">>", currentNode.getContent(), 1);
                        alg.appendMove(m, 1);
                        currentCost++;
                        if (currentNode.isEndOfTrace())
                            break;
                    }
                }

                if (currentCost < optimalCost) {
                    optimalCost = currentCost;
                    optimalAlg = alg;
                    optimalNode = currentNode;
                }
            }

            HashMap<TrieNode, Alignment> optimalLeafAlignment = null;
            int oldestStateMinCost = 9999999;
            for (State s : oldestStates) {
                HashMap<TrieNode, Alignment> currentLeafAlignment = findOptimalLeafNode(s, optimalCost);
                if (currentLeafAlignment != null) {
                    Map.Entry<TrieNode, Alignment> entry = currentLeafAlignment.entrySet().iterator().next();
                    if (entry.getValue().getTotalCost() < oldestStateMinCost) {
                        optimalLeafAlignment = currentLeafAlignment;
                        oldestStateMinCost = entry.getValue().getTotalCost();
                    }
                }
            }

            if (optimalLeafAlignment != null) {
                Map.Entry<TrieNode, Alignment> entry = optimalLeafAlignment.entrySet().iterator().next();
                return new State(entry.getValue(), new ArrayList<>(), entry.getKey(), entry.getValue().getTotalCost());
            } else {
                return new State(optimalAlg, new ArrayList<>(), optimalNode, optimalCost);
            }

        } else if (finalState) {
            // did not find matching ID
            // returning only model moves for shortest path
            TrieNode minNode = modelTrie.getNodeOnShortestTrace();
            TrieNode currentNode = minNode;
            Alignment alg = new Alignment();
            List<Move> moves = new ArrayList<>();
            while (currentNode.getLevel() > 0) {
                Move m = new Move(">>", currentNode.getContent(), 1);
                moves.add(0, m);
                currentNode = currentNode.getParent();
            }
            for (Move m : moves)
                alg.appendMove(m);

            return new State(alg, new ArrayList<>(), minNode, alg.getTotalCost());

        }

        // did not find a matching case ID
        // OR there is no state with most recent decay time and no trace postfix (note: this part should not happen)
        return null;
    }

    public HashMap<TrieNode, Alignment> findOptimalLeafNode(State state, double costLimit) {

        double baseCost = state.getWightedSumOfCosts();
        double cost;
        int lastIndex;
        int checkpointLevel;
        int currentLevel;
        int additionalCost;
        TrieNode currentNode;
        TrieNode optimalNode = null;

        List<String> originalPostfix = state.getTracePostfix();
        List<String> postfix;
        int stateLevel = state.getNode().getLevel();
        int originalPostfixSize = originalPostfix.size();
        int postfixSize;
        int logMovesDone;
        double maxLevel = stateLevel + originalPostfixSize + costLimit - 1;
        List<TrieNode> leaves = modelTrie.getLeavesFromNode(state.getNode(), (int) maxLevel);
        Map<Integer, String> optimalMoves = new HashMap<>();
        Map<Integer, String> moves;

        for (TrieNode n : leaves) {
            //
            if (n.getLevel() > maxLevel) {
                continue; // the level limit has been updated and this leaf can never improve the current optimal cost
            }
            postfix = new ArrayList<>(originalPostfix);
            currentNode = n;
            cost = baseCost;
            checkpointLevel = stateLevel + originalPostfixSize;
            currentLevel = currentNode.getLevel();
            moves = new HashMap<>();
            logMovesDone = 0;
            while (cost < costLimit) {
                lastIndex = postfix.lastIndexOf(currentNode.getContent());
                if (lastIndex < 0) {
                    if (currentLevel > checkpointLevel) {
                        cost += 1;
                        moves.put(moves.size(), "model");
                    } else {
                        cost += 2; // we now need to make both a log and model move
                        moves.put(moves.size(), "model");
                        moves.put(moves.size(), "log");
                        logMovesDone++;
                    }
                } else {

                    postfixSize = postfix.size();
                    additionalCost = (postfixSize - lastIndex) - 1;
                    if (additionalCost > 0) {
                        while (additionalCost > 0) {
                            if (logMovesDone > 0) {
                                logMovesDone--;
                                additionalCost--;
                                if (additionalCost == 0) {
                                    break;
                                }
                            } else {
                                cost++;
                                moves.put(moves.size(), "log");
                                additionalCost--;
                            }
                        }
                    }
                    moves.put(moves.size(), "sync");
                    postfix.subList(lastIndex, postfixSize).clear();
                    checkpointLevel = stateLevel + postfix.size();
                }

                currentNode = currentNode.getParent();
                currentLevel = currentNode.getLevel();
                if (currentLevel == stateLevel & cost < costLimit & postfix.size() == 0) {
                    // handle new cost limit
                    maxLevel = stateLevel + originalPostfixSize + cost - 1;
                    costLimit = state.getWightedSumOfCosts() + cost;
                    optimalNode = n;
                    optimalMoves = moves;
                    break;
                }

                if (currentLevel == stateLevel)
                    break; // cost has not improved but we have reached the original state level
            }
        }

        if (optimalNode == null) {
            return null;
        }
        // calculate alignment
        Alignment alg = new Alignment(state.getAlignment());

        currentNode = optimalNode;
        Move m;
        List<Move> movesForAlg = new ArrayList<>();
        int optimalMovesOrigSize = optimalMoves.size();
        while (optimalMoves.size() > 0) {
            String currentMove = optimalMoves.remove(optimalMovesOrigSize - optimalMoves.size());
            if (currentMove == "model") {
                m = new Move(">>", currentNode.getContent(), 1);
                currentNode = currentNode.getParent();
            } else if (currentMove == "log") {
                m = new Move(originalPostfix.remove(originalPostfix.size() - 1), ">>", 1);
            } else {
                String event = originalPostfix.remove(originalPostfix.size() - 1);
                m = new Move(event, event, 0);
                currentNode = currentNode.getParent();
            }
            movesForAlg.add(0, m);
        }

        for (Move mv : movesForAlg) {
            alg.appendMove(mv);
        }

        HashMap<TrieNode, Alignment> result = new HashMap<>();
        result.put(optimalNode, alg);
        return result;
    }

    protected double getMaxCostOfStates(HashMap<String, State> s) {
        double maxCost = Double.MIN_VALUE;
        Collection<State> states = s.values();
        for (State st : states) {
            if (st.getWightedSumOfCosts() >= maxCost)
                maxCost = st.getWightedSumOfCosts();
        }
        return maxCost;
    }

    protected double updateCost(double currCost, MoveType mv, TrieNode prevEta, TrieNode eta) {
        if (mv == MoveType.LOG_MOVE) {
            return currCost;
        }
        return currCost - prevEta.getScaledConfCost() + eta.getScaledConfCost();
    }

    protected int computeDecayTime() {
        if (discountedDecayTime)
            return Math.max(Math.round(averageTrieLength * decayTimeMultiplier), defaultDecayTime);
        return defaultDecayTime;
    }
}