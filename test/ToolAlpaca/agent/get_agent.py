import os
import json
import logging
from typing import List, Tuple, Any, Union

from pydantic import Field
from langchain import LLMChain
from langchain.agents import ZeroShotAgent
from langchain.schema import AgentAction, AgentFinish

from .tools import Tool, GetDetailsTool, tool_projection
from .custom_parser import CustomMRKLOutputParser
from .custom_agent_executor import CustomAgentExecutor
from utils import load_openapi_spec, escape
from .agent_prompts import train_prompt_v2, test_prompt_v1
from .custom_agent import CustomZeroShotAgent


logger = logging.getLogger(__name__)


def get_agent(
    llm,
    api_data,
    server_url,
    agent_prompt=train_prompt_v2,
    enable_getDetails=True,
    return_intermediate_steps=True,
):
        
    openapi_spec = load_openapi_spec(api_data["Documentation"], replace_refs=True)
    """
openapi
  3.0.1
info
  title
    Nager.Date API - V3
  description
    Nager.Date is open source software. If you would like to support the project you can award a GitHub star ⭐ or much better <a href='https://github.com/sponsors/nager'>start a sponsorship</a>
  contact
    name
      Nager.Date on GitHub
    url
      https://github.com/nager/Nager.Date
  license
    name
      MIT License
    url
      https://github.com/nager/Nager.Date/blob/master/LICENSE.md
  version
    v3
servers
  [{'url': 'https://date.nager.at/'}]
paths
  /api/v3/CountryInfo/{countryCode}
    get
      tags
        ['Country']
      summary
        Get country info for the given country
      operationId
        CountryCountryInfo
      parameters
        [{'name': 'countryCode', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'in': 'path', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}]
      responses
        {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}, 'application/json': {'schema': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}, 'text/json': {'schema': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}}}}
  /api/v3/AvailableCountries
    get
      tags
        ['Country']
      summary
        Get all available countries
      operationId
        CountryAvailableCountries
      responses
        {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}}}}}}
  /api/v3/LongWeekend/{year}/{countryCode}
    get
      tags
        ['LongWeekend']
      summary
        Get long weekends for a given country
      operationId
        LongWeekendLongWeekend
      parameters
        [{'name': 'year', 'in': 'path', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'integer', 'format': 'int32'}}, {'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}]
      responses
        {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}}}}}}
  /api/v3/PublicHolidays/{year}/{countryCode}
    get
      tags
        ['PublicHoliday']
      summary
        Get public holidays
      operationId
        PublicHolidayPublicHolidaysV3
      parameters
        [{'name': 'year', 'in': 'path', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'integer', 'format': 'int32'}}, {'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}]
      responses
        {'200': {'description': 'Public holidays', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}}}, '400': {'description': 'Validation failure'}, '404': {'description': 'CountryCode is unknown'}}
  /api/v3/IsTodayPublicHoliday/{countryCode}
    get
      tags
        ['PublicHoliday']
      summary
        Is today a public holiday
      description
        The calculation is made on the basis of UTC time to adjust the time please use the offset.<br />
        This is a special endpoint for `curl`<br /><br />
        200 = Today is a public holiday<br />
        204 = Today is not a public holiday<br /><br />
        `STATUSCODE=$(curl --silent --output /dev/stderr --write-out "%{http_code}" https://date.nager.at/Api/v2/IsTodayPublicHoliday/AT)`<br /><br />
        `if [ $STATUSCODE -ne 200 ]; then # error handling; fi`
      operationId
        PublicHolidayIsTodayPublicHoliday
      parameters
        [{'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}, {'name': 'countyCode', 'in': 'query', 'required': False, 'style': 'form', 'explode': True, 'schema': {'type': 'string'}}, {'name': 'offset', 'in': 'query', 'description': 'utc timezone offset', 'required': False, 'style': 'form', 'explode': True, 'schema': {'maximum': 12, 'minimum': -12, 'type': 'integer', 'format': 'int32', 'default': 0}}]
      responses
        {'200': {'description': 'Today is a public holiday'}, '204': {'description': 'Today is not a public holiday'}, '400': {'description': 'Validation failure'}, '404': {'description': 'CountryCode is unknown'}}
  /api/v3/NextPublicHolidays/{countryCode}
    get
      tags
        ['PublicHoliday']
      summary
        Returns the upcoming public holidays for the next 365 days for the given country
      operationId
        PublicHolidayNextPublicHolidays
      parameters
        [{'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}]
      responses
        {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}}}}
  /api/v3/NextPublicHolidaysWorldwide
    get
      tags
        ['PublicHoliday']
      summary
        Returns the upcoming public holidays for the next 7 days
      operationId
        PublicHolidayNextPublicHolidaysWorldwide
      responses
        {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}}}}
  /api/v3/Version
    get
      tags
        ['Version']
      summary
        Get version of the used Nager.Date library
      operationId
        VersionGetVersion
      responses
        {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}, 'application/json': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}, 'text/json': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}}}}
components
  schemas
    CountryInfoDto
      type
        object
      properties
        {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {'type': 'object', 'properties': {...}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}}
      additionalProperties
        False
      description
        CountryInfo Dto
    CountryV3Dto
      type
        object
      properties
        {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}
      additionalProperties
        False
      description
        Country
    LongWeekendV3Dto
      type
        object
      properties
        {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}
      additionalProperties
        False
      description
        Long Weekend
    PublicHolidayType
      type
        string
      enum
        ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']
    PublicHolidayV3Dto
      type
        object
      properties
        {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}
      additionalProperties
        False
      description
        Public Holiday
    VersionInfoDto
      type
        object
      properties
        {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}
      additionalProperties
        False
VersionGetVersion

    {'openapi': '3.0.1', 
    'info': {'title': 'Nager.Date API - V3', 'description': "Nager.Date is open source software. If you would like to support the project you can award a GitHub star ⭐ or much better <a href='https://github.com/sponsors/nager'>start a sponsorship</a>", 'contact': {'name': 'Nager.Date on GitHub', 'url': 'https://github.com/nager/Nager.Date'}, 'license': {'name': 'MIT License', 'url': 'https://github.com/nager/Nager.Date/blob/master/LICENSE.md'}, 'version': 'v3'}, 
    'servers': [{'url': 'https://date.nager.at/'}], 
    'paths': 
        {'/api/v3/CountryInfo/{countryCode}': {'get': {'tags': ['Country'], 'summary': 'Get country info for the given country', 'operationId': 'CountryCountryInfo', 'parameters': [{'name': 'countryCode', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'in': 'path', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}], 'responses': {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}, 'application/json': {'schema': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}, 'text/json': {'schema': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}}}}}}}, '/api/v3/AvailableCountries': {'get': {'tags': ['Country'], 'summary': 'Get all available countries', 'operationId': 'CountryAvailableCountries', 'responses': {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}}}}}}}}, '/api/v3/LongWeekend/{year}/{countryCode}': {'get': {'tags': ['LongWeekend'], 'summary': 'Get long weekends for a given country', 'operationId': 'LongWeekendLongWeekend', 'parameters': [{'name': 'year', 'in': 'path', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'integer', 'format': 'int32'}}, {'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}], 'responses': {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}}}}}}}}, '/api/v3/PublicHolidays/{year}/{countryCode}': {'get': {'tags': ['PublicHoliday'], 'summary': 'Get public holidays', 'operationId': 'PublicHolidayPublicHolidaysV3', 'parameters': [{'name': 'year', 'in': 'path', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'integer', 'format': 'int32'}}, {'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}], 'responses': {'200': {'description': 'Public holidays', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}}}, '400': {'description': 'Validation failure'}, '404': {'description': 'CountryCode is unknown'}}}}, '/api/v3/IsTodayPublicHoliday/{countryCode}': {'get': {'tags': ['PublicHoliday'], 'summary': 'Is today a public holiday', 'description': 'The calculation is made on the basis of UTC time to adjust the time please use the offset.<br />\r\nThis is a special endpoint for `curl`<br /><br />\r\n200 = Today is a public holiday<br />\r\n204 = Today is not a public holiday<br /><br />\r\n`STATUSCODE=$(curl --silent --output /dev/stderr --write-out "%{http_code}" https://date.nager.at/Api/v2/IsTodayPublicHoliday/AT)`<br /><br />\r\n`if [ $STATUSCODE -ne 200 ]; then # error handling; fi`', 'operationId': 'PublicHolidayIsTodayPublicHoliday', 'parameters': [{'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}, {'name': 'countyCode', 'in': 'query', 'required': False, 'style': 'form', 'explode': True, 'schema': {'type': 'string'}}, {'name': 'offset', 'in': 'query', 'description': 'utc timezone offset', 'required': False, 'style': 'form', 'explode': True, 'schema': {'maximum': 12, 'minimum': -12, 'type': 'integer', 'format': 'int32', 'default': 0}}], 'responses': {'200': {'description': 'Today is a public holiday'}, '204': {'description': 'Today is not a public holiday'}, '400': {'description': 'Validation failure'}, '404': {'description': 'CountryCode is unknown'}}}}, '/api/v3/NextPublicHolidays/{countryCode}': {'get': {'tags': ['PublicHoliday'], 'summary': 'Returns the upcoming public holidays for the next 365 days for the given country', 'operationId': 'PublicHolidayNextPublicHolidays', 'parameters': [{'name': 'countryCode', 'in': 'path', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'required': True, 'style': 'simple', 'explode': False, 'schema': {'type': 'string'}}], 'responses': {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}}}}}}, '/api/v3/NextPublicHolidaysWorldwide': {'get': {'tags': ['PublicHoliday'], 'summary': 'Returns the upcoming public holidays for the next 7 days', 'operationId': 'PublicHolidayNextPublicHolidaysWorldwide', 'responses': {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'application/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}, 'text/json': {'schema': {'type': 'array', 'items': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}}}}}}}}, '/api/v3/Version': {'get': {'tags': ['Version'], 'summary': 'Get version of the used Nager.Date library', 'operationId': 'VersionGetVersion', 'responses': {'200': {'description': 'Success', 'content': {'text/plain': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}, 'application/json': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}, 'text/json': {'schema': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}}}}}}}, 
    'components': {'schemas': 
        {'CountryInfoDto': {'type': 'object', 'properties': {'commonName': {'type': 'string', 'description': 'CommonName', 'nullable': True}, 'officialName': {'type': 'string', 'description': 'OfficialName', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'Two-character represented country code. For instance, CN or cn represents China.', 'nullable': True}, 'region': {'type': 'string', 'description': 'Region', 'nullable': True}, 'borders': {'type': 'array', 'description': 'Country Borders', 'nullable': True, 'items': {...}}}, 'additionalProperties': False, 'description': 'CountryInfo Dto'}, 'CountryV3Dto': {'type': 'object', 'properties': {'countryCode': {'type': 'string', 'nullable': True}, 'name': {'type': 'string', 'nullable': True}}, 'additionalProperties': False, 'description': 'Country'}, 'LongWeekendV3Dto': {'type': 'object', 'properties': {'startDate': {'type': 'string', 'description': 'StartDate', 'format': 'date-time'}, 'endDate': {'type': 'string', 'description': 'EndDate', 'format': 'date-time'}, 'dayCount': {'type': 'integer', 'description': 'DayCount', 'format': 'int32'}, 'needBridgeDay': {'type': 'boolean', 'description': 'NeedBridgeDay'}}, 'additionalProperties': False, 'description': 'Long Weekend'}, 'PublicHolidayType': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}, 'PublicHolidayV3Dto': {'type': 'object', 'properties': {'date': {'type': 'string', 'description': 'The date', 'format': 'date'}, 'localName': {'type': 'string', 'description': 'Local name', 'nullable': True}, 'name': {'type': 'string', 'description': 'English name', 'nullable': True}, 'countryCode': {'type': 'string', 'description': 'ISO 3166-1 alpha-2', 'nullable': True}, 'fixed': {'type': 'boolean', 'description': 'Is this public holiday every year on the same date'}, 'global': {'type': 'boolean', 'description': 'Is this public holiday in every county (federal state)'}, 'counties': {'type': 'array', 'description': 'ISO-3166-2 - Federal states', 'nullable': True, 'items': {'type': 'string'}}, 'launchYear': {'type': 'integer', 'description': 'The launch year of the public holiday', 'format': 'int32', 'nullable': True}, 'types': {'type': 'array', 'description': 'A list of types the public holiday it is valid', 'nullable': True, 'items': {'type': 'string', 'enum': ['Public', 'Bank', 'School', 'Authorities', 'Optional', 'Observance']}}}, 'additionalProperties': False, 'description': 'Public Holiday'}, 'VersionInfoDto': {'type': 'object', 'properties': {'name': {'type': 'string', 'nullable': True}, 'version': {'type': 'string', 'nullable': True}}, 'additionalProperties': False}}}}
    """
    components_descriptions = escape(api_data["Function_Description"]["components"])
    # def cprint(dicts, space=0):
    #     if space > 3:
    #         print("  "*space +str(dicts))
    #         return
    #     if isinstance(dicts, dict):
    #         for k,v in dicts.items():
    #             print("  "*space + k)
    #             cprint(v, space+1)
    #     else:
    #         print("  "*space + str(dicts))
    #     return
    # cprint(openapi_spec, 0)
    tools = [GetDetailsTool()] if not enable_getDetails else []
    for ext_tool in api_data.get("external_tools", []):
        tools.append(tool_projection[ext_tool]())

    for idx, func_name in enumerate(api_data["Function_Projection"]):
        description = escape(api_data["Function_Description"][func_name])
        if idx == len(api_data["Function_Projection"]) - 1:
            # print(func_name)
            description += components_descriptions
        path, method = api_data["Function_Projection"][func_name]
        tools.append(Tool(
            base_url=server_url + "/" + api_data["Name"] if server_url else None,
            func_name=func_name,
            openapi_spec=openapi_spec,
            path=path,
            method=method,
            description=description,
            retrieval_available="retrieval" in api_data.get("external_tools", [])
        ))

    AgentType = CustomZeroShotAgent if agent_prompt == test_prompt_v1 else ZeroShotAgent

    prompt = AgentType.create_prompt(
        tools, 
        prefix=agent_prompt["prefix"], 
        suffix=agent_prompt["suffix"],
        format_instructions=agent_prompt["format_instructions"],
        input_variables=["input", "agent_scratchpad"]
    )
    """
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions with the help of some tools.\nYou have access to the following tools:\n\nretrievalDataFromFile: Retrieve specified information from a file. This tool can ONLY be used when you recieve the message \'The output is too long. You need to use the \'retrievalDataFromFile\' function to retrieve the output from the file: <file_path>\'.\nParameters: {{"file_path": "Required. String. The path to the file from which information needs to be retrieved.", "query": "Required. String. The specific information to be retrieved from the file."}}\nOutput: The retrieved information from the file.\n - Format: text/json\n - Structure: {{"retrieved_info": "String containing the requested information retrieved from the file."}}\nCountryCountryInfo: Get country info for the given country\nParameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Success.\n - Format: text/plain\n - Structure: #CountryInfoDto\nCountryAvailableCountries: Get all available countries\nParameters: {{}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#CountryV3Dto]\nLongWeekendLongWeekend: Get long weekends for a given country\nParameters: {{"year": "Required. integer. ", "countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#LongWeekendV3Dto]\nPublicHolidayPublicHolidaysV3: Get public holidays\nParameters: {{"year": "Required. integer. ", "countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Public holidays.\n - Format: text/plain\n - Structure: Array[#PublicHolidayV3Dto]\nPublicHolidayIsTodayPublicHoliday: Is today a public holiday\nParameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China.", "countyCode": "string. ", "offset": "integer. utc timezone offset."}}\nOutput: Today is a public holiday.\n - Format: \n - Structure:\nPublicHolidayNextPublicHolidays: Returns the upcoming public holidays for the next 365 days for the given country\nParameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#PublicHolidayV3Dto]\nPublicHolidayNextPublicHolidaysWorldwide: Returns the upcoming public holidays for the next 7 days\nParameters: {{}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#PublicHolidayV3Dto]\nVersionGetVersion: Get version of the used Nager.Date library\nParameters: {{}}\nOutput: Success.\n - Format: text/plain\n - Structure: #VersionInfoDtoThe detailed output format for the tools is outlined below:\n#CountryInfoDto: Object{{commonName, officialName, countryCode, region, borders: Array[#CountryInfoDto]}}\n#CountryV3Dto: Object{{countryCode, name}}\n#LongWeekendV3Dto: Object{{startDate, endDate, dayCount, needBridgeDay}}\n#PublicHolidayType: \n#PublicHolidayV3Dto: Object{{date, localName, name, countryCode, fixed, global, counties: Array[string], launchYear, types: Array[#PublicHolidayType]}}\n#VersionInfoDto: Object{{name, version}}\n\nThe chat follows this format:\nUSER: the user\'s question\nASSISTANT Thought: the assistant\'s inner thought about what to do next \nASSISTANT Action: the action to take, must be one of [retrievalDataFromFile, CountryCountryInfo, CountryAvailableCountries, LongWeekendLongWeekend, PublicHolidayPublicHolidaysV3, PublicHolidayIsTodayPublicHoliday, PublicHolidayNextPublicHolidays, PublicHolidayNextPublicHolidaysWorldwide, VersionGetVersion].\nASSISTANT Action Input: the input for the action, in JSON format.\nASSISTANT Observation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nASSISTANT Thought: summarize the information gathered\nASSISTANT Response: the final response to the user\nUSER: user\'s next question\n...\n\nBegin!\n\nUSER: {input}\nASSISTANT Thought:{agent_scratchpad}
    A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions with the help of some tools.
    You have access to the following tools:

    retrievalDataFromFile: Retrieve specified information from a file. This tool can ONLY be used when you recieve the message 'The output is too long. You need to use the 'retrievalDataFromFile' function to retrieve the output from the file: <file_path>'.
    Parameters: {{"file_path": "Required. String. The path to the file from which information needs to be retrieved.", "query": "Required. String. The specific information to be retrieved from the file."}}
    Output: The retrieved information from the file.
    - Format: text/json
    - Structure: {{"retrieved_info": "String containing the requested information retrieved from the file."}}
    CountryCountryInfo: Get country info for the given country
    Parameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}
    Output: Success.
    - Format: text/plain
    - Structure: #CountryInfoDto
    CountryAvailableCountries: Get all available countries
    Parameters: {{}}
    Output: Success.
    - Format: text/plain
    - Structure: Array[#CountryV3Dto]
    LongWeekendLongWeekend: Get long weekends for a given country
    Parameters: {{"year": "Required. integer. ", "countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}
    Output: Success.
    - Format: text/plain
    - Structure: Array[#LongWeekendV3Dto]
    PublicHolidayPublicHolidaysV3: Get public holidays
    Parameters: {{"year": "Required. integer. ", "countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}
    Output: Public holidays.
    - Format: text/plain
    - Structure: Array[#PublicHolidayV3Dto]
    PublicHolidayIsTodayPublicHoliday: Is today a public holiday
    Parameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China.", "countyCode": "string. ", "offset": "integer. utc timezone offset."}}
    Output: Today is a public holiday.
    - Format: 
    - Structure:
    PublicHolidayNextPublicHolidays: Returns the upcoming public holidays for the next 365 days for the given country
    Parameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}
    Output: Success.
    - Format: text/plain
    - Structure: Array[#PublicHolidayV3Dto]
    PublicHolidayNextPublicHolidaysWorldwide: Returns the upcoming public holidays for the next 7 days
    Parameters: {{}}
    Output: Success.
    - Format: text/plain
    - Structure: Array[#PublicHolidayV3Dto]
    VersionGetVersion: Get version of the used Nager.Date library
    Parameters: {{}}
    Output: Success.
    - Format: text/plain
    - Structure: #VersionInfoDtoThe detailed output format for the tools is outlined below:
    #CountryInfoDto: Object{{commonName, officialName, countryCode, region, borders: Array[#CountryInfoDto]}}
    #CountryV3Dto: Object{{countryCode, name}}
    #LongWeekendV3Dto: Object{{startDate, endDate, dayCount, needBridgeDay}}
    #PublicHolidayType: 
    #PublicHolidayV3Dto: Object{{date, localName, name, countryCode, fixed, global, counties: Array[string], launchYear, types: Array[#PublicHolidayType]}}
    #VersionInfoDto: Object{{name, version}}

    The chat follows this format:
    USER: the user's question
    ASSISTANT Thought: the assistant's inner thought about what to do next 
    ASSISTANT Action: the action to take, must be one of [retrievalDataFromFile, CountryCountryInfo, CountryAvailableCountries, LongWeekendLongWeekend, PublicHolidayPublicHolidaysV3, PublicHolidayIsTodayPublicHoliday, PublicHolidayNextPublicHolidays, PublicHolidayNextPublicHolidaysWorldwide, VersionGetVersion].
    ASSISTANT Action Input: the input for the action, in JSON format.
    ASSISTANT Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    ASSISTANT Thought: summarize the information gathered
    ASSISTANT Response: the final response to the user
    USER: user's next question
    ...

    Begin!

    USER: {input}
    ASSISTANT Thought:{agent_scratchpad}
    
    """
    # print('A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user\'s questions with the help of some tools.\nYou have access to the following tools:\n\nretrievalDataFromFile: Retrieve specified information from a file. This tool can ONLY be used when you recieve the message \'The output is too long. You need to use the \'retrievalDataFromFile\' function to retrieve the output from the file: <file_path>\'.\nParameters: {{"file_path": "Required. String. The path to the file from which information needs to be retrieved.", "query": "Required. String. The specific information to be retrieved from the file."}}\nOutput: The retrieved information from the file.\n - Format: text/json\n - Structure: {{"retrieved_info": "String containing the requested information retrieved from the file."}}\nCountryCountryInfo: Get country info for the given country\nParameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Success.\n - Format: text/plain\n - Structure: #CountryInfoDto\nCountryAvailableCountries: Get all available countries\nParameters: {{}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#CountryV3Dto]\nLongWeekendLongWeekend: Get long weekends for a given country\nParameters: {{"year": "Required. integer. ", "countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#LongWeekendV3Dto]\nPublicHolidayPublicHolidaysV3: Get public holidays\nParameters: {{"year": "Required. integer. ", "countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Public holidays.\n - Format: text/plain\n - Structure: Array[#PublicHolidayV3Dto]\nPublicHolidayIsTodayPublicHoliday: Is today a public holiday\nParameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China.", "countyCode": "string. ", "offset": "integer. utc timezone offset."}}\nOutput: Today is a public holiday.\n - Format: \n - Structure:\nPublicHolidayNextPublicHolidays: Returns the upcoming public holidays for the next 365 days for the given country\nParameters: {{"countryCode": "Required. string. Two-character represented country code. For instance, CN or cn represents China."}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#PublicHolidayV3Dto]\nPublicHolidayNextPublicHolidaysWorldwide: Returns the upcoming public holidays for the next 7 days\nParameters: {{}}\nOutput: Success.\n - Format: text/plain\n - Structure: Array[#PublicHolidayV3Dto]\nVersionGetVersion: Get version of the used Nager.Date library\nParameters: {{}}\nOutput: Success.\n - Format: text/plain\n - Structure: #VersionInfoDtoThe detailed output format for the tools is outlined below:\n#CountryInfoDto: Object{{commonName, officialName, countryCode, region, borders: Array[#CountryInfoDto]}}\n#CountryV3Dto: Object{{countryCode, name}}\n#LongWeekendV3Dto: Object{{startDate, endDate, dayCount, needBridgeDay}}\n#PublicHolidayType: \n#PublicHolidayV3Dto: Object{{date, localName, name, countryCode, fixed, global, counties: Array[string], launchYear, types: Array[#PublicHolidayType]}}\n#VersionInfoDto: Object{{name, version}}\n\nThe chat follows this format:\nUSER: the user\'s question\nASSISTANT Thought: the assistant\'s inner thought about what to do next \nASSISTANT Action: the action to take, must be one of [retrievalDataFromFile, CountryCountryInfo, CountryAvailableCountries, LongWeekendLongWeekend, PublicHolidayPublicHolidaysV3, PublicHolidayIsTodayPublicHoliday, PublicHolidayNextPublicHolidays, PublicHolidayNextPublicHolidaysWorldwide, VersionGetVersion].\nASSISTANT Action Input: the input for the action, in JSON format.\nASSISTANT Observation: the result of the action\n... (this Thought/Action/Action Input/Observation can repeat N times)\nASSISTANT Thought: summarize the information gathered\nASSISTANT Response: the final response to the user\nUSER: user\'s next question\n...\n\nBegin!\n\nUSER: {input}\nASSISTANT Thought:{agent_scratchpad}')
    # print(prompt)
    # breakpoint()
    # logger.info(str(prompt))

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    AgentType.return_values = ["output", "Final Thought"]
    agent = AgentType(llm_chain=llm_chain, allowed_tools=[t.name for t in tools])
    if agent_prompt != test_prompt_v1:
        agent.output_parser = CustomMRKLOutputParser()
    
    agent_executor = CustomAgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=return_intermediate_steps
    )
    return agent_executor

