from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import numpy as np
import pickle
import uvicorn

# Load the Logistic Regression model
with open("model_1_logreg_gridsearch.pkl", "rb") as f:
    logistic_model = pickle.load(f)

# Initialize FastAPI application
app = FastAPI(
    title="DDoS Detection Model API",
    description="API for deploying the Logistic Regression model to make predictions on DDoS dataset.",
    version="1.0"
)

# Mapping of prediction result to attack type description
attack_types = {
    0: "BENIGN",
    1: "DrDoS_NTP",
    2: "DrDoS_SSDP",
    3: "DrDoS_NetBIOS",
    4: "DrDoS_DNS",
    5: "DrDoS_MSSQL",
    6: "DrDoS_UDP",
    7: "DrDoS_LDAP",
    8: "DrDoS_SNMP"
}

# Define the expected input data structure
class PredictionRequest(BaseModel):
    Protocol: Optional[float] = Field(None, description="Protocol used in the flow (e.g., TCP, UDP), identifying the transport layer protocol.")
    Fwd_Packet_Length_Max: Optional[float] = Field(None, description="Maximum length of forward packets, indicating the largest outgoing packet size.")
    Fwd_Packet_Length_Min: Optional[float] = Field(None, description="Minimum length of forward packets, indicating the smallest outgoing packet size.")
    Bwd_Packet_Length_Min: Optional[float] = Field(None, description="Minimum packet length in the backward direction, representing the smallest backward packet size.")
    Bwd_IAT_Total: Optional[float] = Field(None, description="Total inter-arrival time of packets in the backward direction, summing all intervals.")
    Bwd_IAT_Mean: Optional[float] = Field(None, description="Mean inter-arrival time between packets in the backward direction, indicating time gaps from destination to source.")
    Fwd_PSH_Flags: Optional[float] = Field(None, description="Count of forward packets with the PSH flag, representing push operations in outgoing traffic.")
    Fwd_Packets_per_s: Optional[float] = Field(None, description="Rate of forward packets per second, representing the frequency of outbound packets.")
    Min_Packet_Length: Optional[float] = Field(None, description="Minimum packet length, representing the smallest packet size.")
    Max_Packet_Length: Optional[float] = Field(None, description="Maximum packet length, indicating the largest packet size.")
    SYN_Flag_Count: Optional[float] = Field(None, description="Count of packets with the SYN flag, used to initiate TCP connections.")
    PSH_Flag_Count: Optional[float] = Field(None, description="Count of packets with the PSH flag, used to detect push operations in TCP.")
    ACK_Flag_Count: Optional[float] = Field(None, description="Count of packets with the ACK flag, indicating acknowledgement signals in TCP connections.")
    URG_Flag_Count: Optional[float] = Field(None, description="Count of packets with the URG flag, which indicates urgent data in TCP packets.")
    Down_Up_Ratio: Optional[float] = Field(None, description="Ratio of downstream to upstream packets or data volume, often used to detect traffic patterns associated with certain attack types.")
    Init_Win_bytes_backward: Optional[float] = Field(None, description="Initial window bytes in the backward direction, representing starting data size.")
    min_seg_size_forward: Optional[float] = Field(None, description="Minimum segment size in the forward direction, representing smallest forward segment.")

# Define prediction endpoint
@app.post("/predict", tags=["Logistic Regression Model"])
async def predict_logistic(request: PredictionRequest):
    """
    Predict if the traffic is benign or a type of DDoS attack.
    Returns a descriptive string indicating the traffic type.
    """
    # Convert input data to the format expected by the model
    data = np.array([
        request.Protocol, request.Fwd_Packet_Length_Max, request.Fwd_Packet_Length_Min,
        request.Bwd_Packet_Length_Min, request.Bwd_IAT_Total, request.Bwd_IAT_Mean,
        request.Fwd_PSH_Flags, request.Fwd_Packets_per_s, request.Min_Packet_Length,
        request.Max_Packet_Length, request.SYN_Flag_Count, request.PSH_Flag_Count,
        request.ACK_Flag_Count, request.URG_Flag_Count, request.Down_Up_Ratio,
        request.Init_Win_bytes_backward, request.min_seg_size_forward
    ]).reshape(1, -1)  # Reshape for single prediction
    
    # Make prediction
    prediction = logistic_model.predict(data)[0]
    attack_description = attack_types.get(prediction, "Unknown Attack Type")
    
    return {"prediction": attack_description}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)