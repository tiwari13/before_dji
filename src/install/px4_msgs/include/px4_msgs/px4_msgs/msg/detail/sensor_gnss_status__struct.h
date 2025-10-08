// generated from rosidl_generator_c/resource/idl__struct.h.em
// with input from px4_msgs:msg/SensorGnssStatus.idl
// generated code does not contain a copyright notice

// IWYU pragma: private, include "px4_msgs/msg/sensor_gnss_status.h"


#ifndef PX4_MSGS__MSG__DETAIL__SENSOR_GNSS_STATUS__STRUCT_H_
#define PX4_MSGS__MSG__DETAIL__SENSOR_GNSS_STATUS__STRUCT_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Constants defined in the message

/// Struct defined in msg/SensorGnssStatus in the package px4_msgs.
/**
  * Gnss quality indicators
 */
typedef struct px4_msgs__msg__SensorGnssStatus
{
  /// time since system start (microseconds)
  uint64_t timestamp;
  /// unique device ID for the sensor that does not change between power cycles
  uint32_t device_id;
  /// Set to true if quality indicators are available
  bool quality_available;
  /// Corrections quality from 0 to 10, or 255 if not available
  uint8_t quality_corrections;
  /// Overall receiver operating status from 0 to 10, or 255 if not available
  uint8_t quality_receiver;
  /// Quality of GNSS signals from 0 to 10, or 255 if not available
  uint8_t quality_gnss_signals;
  /// Expected post processing quality from 0 to 10, or 255 if not available
  uint8_t quality_post_processing;
} px4_msgs__msg__SensorGnssStatus;

// Struct for a sequence of px4_msgs__msg__SensorGnssStatus.
typedef struct px4_msgs__msg__SensorGnssStatus__Sequence
{
  px4_msgs__msg__SensorGnssStatus * data;
  /// The number of valid items in data
  size_t size;
  /// The number of allocated items in data
  size_t capacity;
} px4_msgs__msg__SensorGnssStatus__Sequence;

#ifdef __cplusplus
}
#endif

#endif  // PX4_MSGS__MSG__DETAIL__SENSOR_GNSS_STATUS__STRUCT_H_
