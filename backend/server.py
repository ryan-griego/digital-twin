def call_bedrock(conversation: List[Dict], user_message: str) -> str:
    """Call AWS Bedrock with conversation history"""

    # Build messages in Bedrock format for Nova models
    messages = []

    # Add system prompt as the first message (Amazon Nova models support this)
    messages.append({
        "role": "user",
        "content": [{"text": f"System: {prompt()}"}]
    })

    # Add a simulated assistant acknowledgment to establish the pattern
    messages.append({
        "role": "assistant",
        "content": [{"text": "Understood. I am Ryan's digital twin and will respond helpfully based on the system information provided."}]
    })

    # Add conversation history (ensure proper alternation)
    for msg in conversation[-10:]:  # Limit to last 10 messages
        if msg["role"] in ["user", "assistant"]:
            messages.append({
                "role": msg["role"],
                "content": [{"text": msg["content"]}]
            })

    # Add current user message
    messages.append({
        "role": "user",
        "content": [{"text": user_message}]
    })

    try:
        # Call Bedrock using the converse API
        response = bedrock_client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=messages,
            inferenceConfig={
                "maxTokens": 1500,
                "temperature": 0.7,
                "topP": 0.9
            }
        )

        # Extract the response text - correct path for converse API
        return response["output"]["message"]["content"][0]["text"]

    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == 'ValidationException':
            raise HTTPException(status_code=400, detail="Invalid message format for Bedrock")
        elif error_code == 'AccessDeniedException':
            raise HTTPException(status_code=403, detail="Access denied to Bedrock model")
        else:
            raise HTTPException(status_code=500, detail=f"Bedrock error: {str(e)}")
    except Exception as e:
        # Catch any other exceptions
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
