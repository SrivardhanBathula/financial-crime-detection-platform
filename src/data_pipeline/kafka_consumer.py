"""
Real-Time Financial Transaction Kafka Consumer
Ingests streaming transaction data for fraud detection pipeline.
"""

import json
import logging
from typing import Callable, Optional
from dataclasses import dataclass, field
from datetime import datetime

from confluent_kafka import Consumer, KafkaError, KafkaException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransactionEvent:
    transaction_id: str
    account_id: str
    amount: float
    currency: str
    merchant_id: str
    merchant_category: str
    timestamp: datetime
    location: dict
    device_fingerprint: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "TransactionEvent":
        return cls(
            transaction_id=data["transaction_id"],
            account_id=data["account_id"],
            amount=float(data["amount"]),
            currency=data.get("currency", "USD"),
            merchant_id=data["merchant_id"],
            merchant_category=data.get("merchant_category", "UNKNOWN"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            location=data.get("location", {}),
            device_fingerprint=data.get("device_fingerprint"),
            metadata=data.get("metadata", {}),
        )


class FinancialTransactionConsumer:
    """
    Kafka consumer for real-time financial transaction ingestion.
    Handles deserialization, validation, and downstream routing.
    """

    def __init__(
        self,
        bootstrap_servers: str,
        group_id: str,
        topics: list[str],
        auto_offset_reset: str = "latest",
    ):
        self.config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
            "enable.auto.commit": False,
            "max.poll.interval.ms": 300000,
            "session.timeout.ms": 30000,
        }
        self.topics = topics
        self.consumer = Consumer(self.config)
        self._running = False

    def start(self, handler: Callable[[TransactionEvent], None], batch_size: int = 100):
        """
        Start consuming messages and invoke handler for each transaction batch.
        """
        self.consumer.subscribe(self.topics)
        self._running = True
        logger.info(f"Consumer started. Subscribed to: {self.topics}")

        batch: list[TransactionEvent] = []

        try:
            while self._running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug(f"End of partition: {msg.topic()} [{msg.partition()}]")
                    else:
                        raise KafkaException(msg.error())
                    continue

                try:
                    raw = json.loads(msg.value().decode("utf-8"))
                    event = TransactionEvent.from_dict(raw)
                    batch.append(event)

                    if len(batch) >= batch_size:
                        handler(batch)
                        self.consumer.commit(asynchronous=False)
                        logger.info(f"Committed batch of {len(batch)} transactions.")
                        batch = []

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Skipping malformed message: {e}")

        except KeyboardInterrupt:
            logger.info("Consumer interrupted by user.")
        finally:
            if batch:
                handler(batch)
                self.consumer.commit(asynchronous=False)
            self.consumer.close()
            logger.info("Consumer closed.")

    def stop(self):
        self._running = False


def process_transaction_batch(events: list[TransactionEvent]):
    """
    Example handler: route batch to feature engineering and scoring pipeline.
    Replace with actual pipeline integration.
    """
    logger.info(f"Processing {len(events)} transactions...")
    for event in events:
        logger.debug(f"  TXN {event.transaction_id} | ${event.amount} | {event.merchant_category}")


if __name__ == "__main__":
    consumer = FinancialTransactionConsumer(
        bootstrap_servers="localhost:9092",
        group_id="fraud-detection-group",
        topics=["financial_transactions", "wire_transfers"],
    )
    consumer.start(handler=process_transaction_batch, batch_size=500)
