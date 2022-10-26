package ee.ut.cs.dsg.confcheck.util;

import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

public class PredictionsClient {

    private final HashMap<String, String> urls;
    private HttpURLConnection con = null;

    public PredictionsClient(HashMap<String, String> urls) {
        this.urls = urls;
    }

    public int initModel(String urlKey, Map<String, String> params) {
        String enc_params;
        try {
            enc_params = getParamsString(params);
        } catch (UnsupportedEncodingException e) {
            throw new RuntimeException(e);
        }

        try {
            con = createConnection(urls.get(urlKey) + enc_params);
            return con.getResponseCode();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private String getParamsString(Map<String, String> params) throws UnsupportedEncodingException {
        StringBuilder result = new StringBuilder();
        result.append("?");

        for (Map.Entry<String, String> entry : params.entrySet()) {
            result.append(entry.getKey());
            result.append("=");
            result.append(entry.getValue());
            result.append("&");
        }

        String resultString = result.toString();
        return resultString.length() > 0 ? resultString.substring(0, resultString.length() - 1) : resultString;
    }

    public float getPrefixProb(String urlKey, String jsonInputString) {
        int status;
        float retVal = 0;

        try {
            con = createConnection(urls.get(urlKey));
            con.setRequestMethod("POST");
            con.setRequestProperty("Content-Type", "application/json");
            con.setDoOutput(true);

            try (OutputStream os = con.getOutputStream()) {
                byte[] input = jsonInputString.getBytes(StandardCharsets.UTF_8);
                os.write(input, 0, input.length);
            }
            status = con.getResponseCode();
            if (status == 200) {
                Reader reader = null;
                try {
                    reader = new InputStreamReader(con.getInputStream(), StandardCharsets.UTF_8);
                    try (BufferedReader br = new BufferedReader(reader)) {
                        StringBuilder response = new StringBuilder();
                        String responseLine;
                        while ((responseLine = br.readLine()) != null) {
                            response.append(responseLine.trim());
                        }
                        retVal = Float.parseFloat(response.toString());
                    }
                } finally {
                    if (reader != null) try {
                        reader.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            } else {
                System.out.printf("Status code: %s%n", status);
                System.out.printf("Payload: %s", jsonInputString);
                getPrefixProb(urlKey, jsonInputString);
            }
        } catch (NullPointerException | IOException e) {
            e.printStackTrace();
        }
        return retVal;
    }

    private HttpURLConnection createConnection(String endpoint) {
        try {
            URL url = new URL(endpoint);
            con = (HttpURLConnection) url.openConnection();
        } catch (java.net.MalformedURLException e) {
            e.printStackTrace();
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return con;
    }

    public void closeConnection() {
        if (con != null) {
            con.disconnect();
            con = null;
        }
    }
}
