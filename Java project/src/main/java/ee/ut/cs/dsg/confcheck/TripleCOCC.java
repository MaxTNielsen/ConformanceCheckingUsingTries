package ee.ut.cs.dsg.confcheck;

import ee.ut.cs.dsg.confcheck.alignment.Alignment;
import ee.ut.cs.dsg.confcheck.alignment.Move;
import ee.ut.cs.dsg.confcheck.cost.CostFunction;
import ee.ut.cs.dsg.confcheck.cost.DualProgressiveCostFunction;
import ee.ut.cs.dsg.confcheck.trie.Trie;
import ee.ut.cs.dsg.confcheck.trie.TrieNode;
import ee.ut.cs.dsg.confcheck.util.Configuration.MoveType;

import java.util.*;

public class TripleCOCC extends ConformanceChecker {

    protected boolean verbose = false;
    protected final CostFunction costFunction;
    final protected HashMap<String, TreeMap<Integer, TrieNode>> warmStartMap = modelTrie.getWarmStart();

    // Streaming variables
    protected boolean replayWithLogMoves = true;
    protected int minDecayTime = 3;
    protected float decayTimeMultiplier = 0.3F;
    protected boolean discountedDecayTime = true; // if set to false then uses fixed minDecayTime value
    protected int averageTrieLength = 0;

    protected boolean isStandardAlign;


    public TripleCOCC(Trie trie, int logCost, int modelCost, int maxStatesInQueue, int maxTrials, CostFunction costFunction, boolean isStandardAlign) {
        super(trie, logCost, modelCost, maxStatesInQueue);
        rnd = new Random(19);
        this.maxTrials = maxTrials;
        inspectedLogTraces = new Trie(trie.getMaxChildren());
        this.costFunction = costFunction;
        this.isStandardAlign = isStandardAlign;

        if (discountedDecayTime) {
            this.averageTrieLength = trie.getAvgTraceLength();
        }

        if (!this.isStandardAlign) {
            modelTrie.computeConfidenceCostForAllNodes("avg");
            modelTrie.computeScaledConfidenceCost(modelTrie.getRoot());
        }
    }

    public TripleCOCC(Trie trie, int logCost, int modelCost, int maxStatesInQueue, boolean isStandardAlign) {
        this(trie, logCost, modelCost, maxStatesInQueue, 10000, isStandardAlign);
    }

    public TripleCOCC(Trie trie, int logCost, int modelCost, int maxStatesInQueue, int maxTrials, boolean isStandardAlign) {
        this(trie, logCost, modelCost, maxStatesInQueue, maxTrials, new DualProgressiveCostFunction(), isStandardAlign);
    }

    public Alignment check(List<String> trace) {
        System.out.println("Only implemented for compatibility with interface");
        return new Alignment();
    }

    public HashMap<String, State> check(List<String> trace, String caseId) {

        traceSize = trace.size();
        State state;
        State previousState;
        StatesBuffer caseStatesInBuffer = null;
        Alignment alg;
        List<String> traceSuffix;
        HashMap<String, State> currentStates = new HashMap<>();
        ArrayList<State> syncMoveStates = new ArrayList<>();

        double boundedCost;
        double minCost = isStandardAlign ? 1.0 : 2.0;

        // iterate over the trace - choose event by event
        // modify everything into accepting event instead of list of events

        if (statesInBuffer.containsKey(caseId)) {
            // case exists, fetch last state
            caseStatesInBuffer = statesInBuffer.get(caseId);
            currentStates = caseStatesInBuffer.getCurrentStates();
            boundedCost = Math.max(getMaxCostOfStates(currentStates) + 1, minCost);

        } else {
            // if sync move(s) --> add sync move(s) to currentStates. If one of the moves will not be sync move, then start checking from that move.
            boundedCost = minCost;
            currentStates.put(new Alignment().toString(), new State(new Alignment(), new ArrayList<String>(), modelTrie.getRoot(), 0.0, computeDecayTime(new Alignment()) + 1)); // larger decay time because this is decremented in this iteration
        }

        for (String event : trace) {
            // sync moves
            // we iterate over all states
            for (Iterator<Map.Entry<String, State>> states = currentStates.entrySet().iterator(); states.hasNext(); ) {

                Map.Entry<String, State> entry = states.next();
                previousState = entry.getValue();
                if (previousState.getTracePostfix().size() != 0) {
                    continue; // we are not interested in previous states which already have a suffix (i.e. they already are non-synchronous)
                }

                state = checkForSyncMoves(event, previousState);
                if (state != null) {
                    // we are only interested in new states which are synced (i.e. they do not have a suffix)
                    syncMoveStates.add(state);
                }
            }

            // check if sync moves --> if yes, add sync states, update old states, remove too old states
            if (syncMoveStates.size() > 0) {
                for (Iterator<Map.Entry<String, State>> states = currentStates.entrySet().iterator(); states.hasNext(); ) {
                    Map.Entry<String, State> entry = states.next();
                    previousState = entry.getValue();
                    int previousDecayTime = previousState.getDecayTime();
                    // remove states with decayTime less than 2
                    if (previousDecayTime < 2) {
                        states.remove();
                    } else {
                        List<String> postfix = new ArrayList<>();
                        postfix.add(event);
                        previousState.addTracePostfix(postfix);
                        previousState.setDecayTime(previousDecayTime - 1);
                    }
                }

                for (State s : syncMoveStates) {
                    alg = s.getAlignment();
                    currentStates.put(alg.toString(), s);
                }

                syncMoveStates.clear();
                continue;
            }

            // no sync moves. We iterate over the states, trying to make model and log moves
            HashMap<String, State> statesToIterate = new HashMap<>(currentStates);
            List<State> interimCurrentStates = new ArrayList<>();
            List<String> traceEvent = new ArrayList<>();
            traceEvent.add(event);

            double currentMinCost = 99999;

            for (Map.Entry<String, State> entry : statesToIterate.entrySet()) {
                previousState = entry.getValue();
                traceSuffix = previousState.getTracePostfix();

                //if (previousState.getWeightedSumOfCosts() + traceSuffix.size() + 1 <= boundedCost) {
                State logMoveState = handleLogMove(traceEvent, previousState, "");
                interimCurrentStates.add(logMoveState);
                //}

                traceSuffix.addAll(traceEvent);

                //if (previousState.getWeightedSumOfCosts() + 1 <= boundedCost) {
                List<State> modelMoveStates = handleModelMoves(traceSuffix, previousState, null);
                   /* for (State mm : modelMoveStates) {
                        if (mm.getWeightedSumOfCosts() <= boundedCost)
                            interimCurrentStates.add(mm);
                    }
                }*/

                // add log move
                if (logMoveState.getWeightedSumOfCosts() < currentMinCost) {
                    interimCurrentStates.clear();
                    interimCurrentStates.add(logMoveState);
                    currentMinCost = logMoveState.getWeightedSumOfCosts();
                } else if (logMoveState.getWeightedSumOfCosts() == currentMinCost) {
                    interimCurrentStates.add(logMoveState);
                }

                // add model moves
                currentMinCost = getCurrentMinCost(interimCurrentStates, currentMinCost, modelMoveStates);

                //List<State> warmStartMoves = handleWarmStartMove(traceEvent, previousState, currentMinCost);

                List<State> warmStartMoves = new ArrayList<>();

                //if (previousState.getAlignment().getTraceSize() == 0) {
                warmStartMoves = handleWarmStartMove(traceEvent, previousState, currentMinCost);
                //}

                /*for (State wm : warmStartMoves) {
                    if (wm.getWeightedSumOfCosts() <= boundedCost)
                        interimCurrentStates.add(wm);
                }*/

                currentMinCost = getCurrentMinCost(interimCurrentStates, currentMinCost, warmStartMoves);

                int previousStateDecayTime = previousState.getDecayTime();
                if (previousStateDecayTime < 2) {
                    currentStates.remove(previousState.getAlignment().toString());
                } else {
                    previousState.setDecayTime(previousStateDecayTime - 1);
                }
            }

            // add new states with the lowest cost
            for (State s : interimCurrentStates) {
                if (s.getWeightedSumOfCosts() == currentMinCost) {
                    currentStates.put(s.getAlignment().toString(), s);
                }
                //currentStates.put(s.getAlignment().toString(), s);
            }
        }

        if (caseStatesInBuffer == null) {
            caseStatesInBuffer = new StatesBuffer(currentStates);
        } else {
            caseStatesInBuffer.setCurrentStates(currentStates);
        }

        statesInBuffer.put(caseId, caseStatesInBuffer);
        return currentStates;

    }

    protected List<State> handleModelMoves(List<String> traceSuffix, State state, State dummyState) {
        TrieNode matchNode;
        Alignment alg;
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
                double cost = state.getWeightedSumOfCosts();
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
                for (Move m : moves) {
                    alg.appendMove(m);
                }
                matchingStates.add(new State(alg, new ArrayList<>(), n, n.getScaledConfCost() + cost, computeDecayTime(alg), state.getCompletenessCost()));
            }
        }
        return matchingStates;
    }

    @Override
    protected State handleLogMove(List<String> traceSuffix, State state, String event) {
        Alignment alg = new Alignment(state.getAlignment());
        State logMoveState;
        List<String> suffix = new ArrayList<>(state.getTracePostfix());
        suffix.addAll(traceSuffix);
        for (String e : suffix) {
            Move logMove = new Move(e, ">>", 1);
            alg.appendMove(logMove);
        }
        logMoveState = new State(alg, new ArrayList<String>(), state.getNode(), updateCost(state.getWeightedSumOfCosts() + suffix.size(), MoveType.LOG_MOVE, state.getNode(), state.getNode()), computeDecayTime(alg), state.getCompletenessCost());
        return logMoveState;
    }

    public State checkForSyncMoves(String event, State currentState) {

        TrieNode prev = currentState.getNode();
        TrieNode node;
        Alignment alg = new Alignment(currentState.getAlignment());
        Move syncMove;

        node = prev.getChild(event);
        if (node == null) {
            return null;
        } else {
            syncMove = new Move(event, event, 0);
            alg.appendMove(syncMove);
            prev = node;
            return new State(alg, new ArrayList<>(), prev, updateCost(currentState.getWeightedSumOfCosts(), MoveType.SYNCHRONOUS_MOVE, prev.getParent(), prev), currentState, computeDecayTime(alg), currentState.getCompletenessCost());
        }
    }

    public List<State> handleWarmStartMove(List<String> event, State state, double boundedCost) {
        List<State> warmStartStates = new ArrayList<>();
        List<String> suffix = state.getTracePostfix();
        int prefixLength = suffix.size();
        suffix.addAll(event);
        TreeMap<Integer, TrieNode> warmStartNodes = warmStartMap.get(suffix.get(0));
        suffix.remove(0);

        if (warmStartNodes != null) {
            for (Map.Entry<Integer, TrieNode> entry : warmStartNodes.entrySet()) {
                int completenessCost = entry.getKey();
                TrieNode warmStartNode = entry.getValue();

                // we only consider warm-start moves that are within the bounded cost
                if (completenessCost + prefixLength <= boundedCost) {
                    Alignment a = new Alignment();
                    a.appendMove(new Move(warmStartNode.getContent(), warmStartNode.getContent(), 0));

                    // we attempt to make synchronous moves on the suffix of the warm-start trace
                    TrieNode fromNode = warmStartNode;
                    Alignment syncAlign = new Alignment(a);
                    for (String activity : suffix) {
                        TrieNode toNode = fromNode.getChild(activity);
                        Move m;
                        if (toNode != null) {
                            m = new Move(toNode.getContent(), toNode.getContent(), 0);
                            syncAlign.appendMove(m);
                            fromNode = toNode;
                            continue;
                        }
                        break;
                    }

                    // test if we have made a full match on suffix
                    if (syncAlign.getMoves().size() - 1 == suffix.size()) {
                        State syncState = new State(syncAlign, new ArrayList<>(), fromNode, updateCost(completenessCost + prefixLength + fromNode.getScaledConfCost(),
                                MoveType.SYNCHRONOUS_MOVE, fromNode, fromNode), computeDecayTime(syncAlign), completenessCost + prefixLength);
                        warmStartStates.add(syncState);
                        break;
                    }

                    // again, we check if the maximum cost will be exceeded by computing warm-start log or model prefix-alignments
                    State s = new State(a, suffix, warmStartNode,
                            updateCost(completenessCost + prefixLength + warmStartNode.getScaledConfCost(), MoveType.SYNCHRONOUS_MOVE, warmStartNode, warmStartNode), computeDecayTime(a), completenessCost + prefixLength);

                    // compute log move state from warm-start node
                    //if (completenessCost + suffix.size() <= boundedCost) {
                    State logMove = handleLogMove(new ArrayList<>(), s, "");
                    warmStartStates.add(logMove);
                    //}

                    // compute model move state from warm-start node
                    //if (completenessCost + 1 <= boundedCost) {
                    warmStartStates.addAll(handleModelMoves(suffix, s, null));
                    //}
                }
            }
        }
        return warmStartStates;
    }

    public State getCurrentOptimalState(String caseId, boolean finalState) { //
        StatesBuffer caseStatesInBuffer;
        HashMap<String, State> currentStates;
        List<State> statesList = new ArrayList<>();
        double currentCost;
        int decayTime;
        State newestState = null;
        State oldestState = null;
        List<State> newestStates = new ArrayList<>();
        List<State> oldestStates = new ArrayList<>();
        List<State> statesToReturn = new ArrayList<>();
        if (statesInBuffer.containsKey(caseId)) {
            caseStatesInBuffer = statesInBuffer.get(caseId);
            currentStates = caseStatesInBuffer.getCurrentStates();
            statesList.addAll(currentStates.values());
            for (State s : statesList) {
                if (finalState) {

                    // if it is end of trace and end of model, then return that state immediately
                    if (
                            ((s.getTracePostfix().size() + s.getNode().getMinPathLengthToEnd()) == 0 ||
                                    (s.getTracePostfix().size() == 0 && s.getNode().isEndOfTrace()))
                                    && s.getWeightedSumOfCosts() == 0
                    ) {
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
                    decayTime = computeDecayTime(s.getAlignment());
                    if (s.getDecayTime() == decayTime & s.getTracePostfix().size() == 0) {
                        //statesToReturn.add(s);
                        return s;
                    }
                }
            }

            // use when confidence is meaningful
            /*if (!finalState) {
                double minCost = Double.MAX_VALUE;
                State bestState = null;
                for (State st : statesToReturn) {
                    if (st.getWeightedSumOfCosts() <= minCost) {
                        minCost = st.getWeightedSumOfCosts();
                        bestState = st;
                    }
                }
                return bestState;
            }*/

            // calculate cost from newestState
            double optimalCost = 9999999;
            Alignment optimalAlg = null;
            TrieNode optimalNode = null;

            for (State s : newestStates) {
                currentCost = s.getWeightedSumOfCosts();

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

        double baseCost = state.getWeightedSumOfCosts();
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
                    costLimit = state.getWeightedSumOfCosts() + cost;
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

    private double getCurrentMinCost(List<State> interimCurrentStates, double currentMinCost, List<State> states) {
        for (State s : states) {
            if (s.getWeightedSumOfCosts() < currentMinCost) {
                interimCurrentStates.clear();
                interimCurrentStates.add(s);
                currentMinCost = s.getWeightedSumOfCosts();
            } else if (s.getWeightedSumOfCosts() == currentMinCost) {
                interimCurrentStates.add(s);
            }
        }

        return currentMinCost;
    }

    protected double updateCost(double currCost, MoveType mv, TrieNode prevEta, TrieNode eta) {
        if (mv == MoveType.LOG_MOVE) {
            return currCost;
        }
        return currCost - prevEta.getScaledConfCost() + eta.getScaledConfCost();
    }

    protected int computeDecayTime(Alignment alg) {
        if (discountedDecayTime)
            return Math.max(Math.round((averageTrieLength - alg.getTraceSize()) * decayTimeMultiplier), minDecayTime);
        return minDecayTime;
    }

    protected double getMaxCostOfStates(HashMap<String, State> s) {
        double maxCost = Double.MIN_VALUE;
        Collection<State> states = s.values();
        for (State st : states) {
            if (st.getWeightedSumOfCosts() >= maxCost)
                maxCost = st.getWeightedSumOfCosts();
        }
        return maxCost;
    }
}