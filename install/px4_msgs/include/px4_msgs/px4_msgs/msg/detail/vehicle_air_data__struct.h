﻿// NOLINT: This file starts with a BOM since it contain non-ASCII characters
// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/VehicleAirData.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "px4_msgs/msg/vehicle_air_data.h"


#ifndef PX4_MSGS__MSG__DETAIL__VEHICLE_AIR_DATA__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__VEHICLE_AIR_DATA__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/VehicleAirData in the package px4_msgs.
/**
  * Vehicle air data
  *
  * Data from the currently selected barometer (plus ambient temperature from the source specified in temperature_source).
  * Includes calculated data such as barometric altitude and air density.
 */
typedef struct px4_msgs__msg__VehicleAirData
{
  /// Time since system start
  uint64_t timestamp;
  /// Timestamp of the raw data
  uint64_t timestamp_sample;
  /// Unique device ID for the selected barometer
  uint32_t baro_device_id;
  /// [m] [@frame MSL] Altitude above MSL calculated from temperature compensated baro sensor data using an ISA corrected for sea level pressure SENS_BARO_QNH
  float baro_alt_meter;
  /// Absolute pressure
  float baro_pressure_pa;
  /// Ambient temperature
  float ambient_temperature;
  /// Source of temperature data: 0: Default Temperature (15°C), 1: External Baro, 2: Airspeed
  uint8_t temperature_source;
  /// Air density
  float rho;
  /// Calibration changed counter. Monotonically increases whenever calibration changes.
  uint8_t calibration_count;
} px4_msgs__msg__VehicleAirData;

// Struct for a sequence of px4_msgs__msg__VehicleAirData.
typedef struct px4_msgs__msg__VehicleAirData__Sequence
{
  px4_msgs__msg__VehicleAirData * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__VehicleAirData__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__VEHICLE_AIR_DATA__STRUCT_H_
