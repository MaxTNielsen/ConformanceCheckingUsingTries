//
// Source code recreated from a .class file by IntelliJ IDEA
// (powered by FernFlower decompiler)
//

import beamline.events.BEvent;
import beamline.models.responses.Response;
import ee.ut.cs.dsg.confcheck.util.AlphabetService;

public class OnlineConformanceResults extends Response {
    private static final long serialVersionUID = 6895821762421722787L;
    private int conformance = 0;
    private int completeness = 0;
    private Double confidence = 0.0;
    private BEvent lastEvent = null;
    private Double processingTime = 0.0;
    private Long memorySizeCases = 0L;
    private Long memorySizeTraces = 0L;
    private int totalStates;
    private int totalCases;
    private int algSize;
    private AlphabetService service;

    public OnlineConformanceResults(AlphabetService service) {
        this.service = service;
    }

    public int getConformance() {
        return this.conformance;
    }

    public void setConformance(int conformance) {
        this.conformance = conformance;
    }

    public int getCompleteness() {
        return this.completeness;
    }

    public void setCompleteness(int completeness) {
        this.completeness = completeness;
    }

    public Double getConfidence() {
        return this.confidence;
    }

    public void setConfidence(Double confidence) {
        this.confidence = confidence;
    }


    public BEvent getLastEvent() {
        return this.lastEvent;
    }

    public void setLastEvent(BEvent lastEvent) {
        this.lastEvent = lastEvent;
    }

    public void setProcessingTime(double l) {
        this.processingTime = l;
    }

    public double getProcessingTime() {
        return this.processingTime;
    }

    public void setTotalStates(int t){
        this.totalStates = t;
    }

    public void setTotalCases(int t){
        this.totalCases = t;
    }

    public int getTotalStates(){
        return this.totalStates;
    }

    public int getTotalCases(){
        return this.totalCases;
    }

    public String toString() {
        String var10000 = this.lastEvent.getTraceName();
        return String.format("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",var10000, service.deAlphabetize(this.lastEvent.getEventName().toCharArray()[0]), this.getConformance(), this.getConfidence(), this.getCompleteness(), this.getTotalCases(), this.getTotalStates(), this.getAlgSize(), this.getProcessingTime(), this.getMemorySizeCases(), this.getMemorySizeTraces());
    }

    public int getAlgSize() {
        return algSize;
    }

    public void setAlgSize(int algSize) {
        this.algSize = algSize;
    }

    public long getMemorySizeCases(){
        return this.memorySizeCases;
    }

    public void setMemorySizeCases(Long m){
        this.memorySizeCases = m;
    }
    public long getMemorySizeTraces(){
        return this.memorySizeTraces;
    }

    public void setMemorySizeTraces(Long m){
        this.memorySizeTraces = m;
    }
}
