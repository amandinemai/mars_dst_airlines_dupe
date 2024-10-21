from pydantic import BaseModel

class TimeStatus(BaseModel):
    Definition: str

class ScheduledTimeUTC(BaseModel):
    DateTime: str

class Departure(BaseModel):
    AirportCode: str
    ScheduledTimeUTC: ScheduledTimeUTC
    TimeStatus: TimeStatus

class Arrival(BaseModel):
    AirportCode: str
    ScheduledTimeUTC: ScheduledTimeUTC
    TimeStatus: TimeStatus

class MarketingCarrier(BaseModel):
    FlightNumber: str

class Flight(BaseModel):
    Departure: Departure
    Arrival: Arrival
    MarketingCarrier: MarketingCarrier