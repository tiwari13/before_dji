// generated from rosidl_typesupport_fastrtps_c/resource/idl__type_support_c.cpp.em
// with input from px4_msgs:msg/RoverSpeedStatus.idl
// generated code does not contain a copyright notice
#include "px4_msgs/msg/detail/rover_speed_status__rosidl_typesupport_fastrtps_c.h"


#include <cassert>
#include <cstddef>
#include <limits>
#include <string>
#include "rosidl_typesupport_fastrtps_c/identifier.h"
#include "rosidl_typesupport_fastrtps_c/serialization_helpers.hpp"
#include "rosidl_typesupport_fastrtps_c/wstring_conversion.hpp"
#include "rosidl_typesupport_fastrtps_cpp/message_type_support.h"
#include "px4_msgs/msg/rosidl_typesupport_fastrtps_c__visibility_control.h"
#include "px4_msgs/msg/detail/rover_speed_status__struct.h"
#include "px4_msgs/msg/detail/rover_speed_status__functions.h"
#include "fastcdr/Cdr.h"

#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-parameter"
# ifdef __clang__
#  pragma clang diagnostic ignored "-Wdeprecated-register"
#  pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
# endif
#endif
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif

// includes and forward declarations of message dependencies and their conversion functions

#if defined(__cplusplus)
extern "C"
{
#endif


// forward declare type support functions


using _RoverSpeedStatus__ros_msg_type = px4_msgs__msg__RoverSpeedStatus;


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
bool cdr_serialize_px4_msgs__msg__RoverSpeedStatus(
  const px4_msgs__msg__RoverSpeedStatus * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: timestamp
  {
    cdr << ros_message->timestamp;
  }

  // Field name: measured_speed_body_x
  {
    cdr << ros_message->measured_speed_body_x;
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    cdr << ros_message->adjusted_speed_body_x_setpoint;
  }

  // Field name: pid_throttle_body_x_integral
  {
    cdr << ros_message->pid_throttle_body_x_integral;
  }

  // Field name: measured_speed_body_y
  {
    cdr << ros_message->measured_speed_body_y;
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    cdr << ros_message->adjusted_speed_body_y_setpoint;
  }

  // Field name: pid_throttle_body_y_integral
  {
    cdr << ros_message->pid_throttle_body_y_integral;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
bool cdr_deserialize_px4_msgs__msg__RoverSpeedStatus(
  eprosima::fastcdr::Cdr & cdr,
  px4_msgs__msg__RoverSpeedStatus * ros_message)
{
  // Field name: timestamp
  {
    cdr >> ros_message->timestamp;
  }

  // Field name: measured_speed_body_x
  {
    cdr >> ros_message->measured_speed_body_x;
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    cdr >> ros_message->adjusted_speed_body_x_setpoint;
  }

  // Field name: pid_throttle_body_x_integral
  {
    cdr >> ros_message->pid_throttle_body_x_integral;
  }

  // Field name: measured_speed_body_y
  {
    cdr >> ros_message->measured_speed_body_y;
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    cdr >> ros_message->adjusted_speed_body_y_setpoint;
  }

  // Field name: pid_throttle_body_y_integral
  {
    cdr >> ros_message->pid_throttle_body_y_integral;
  }

  return true;
}  // NOLINT(readability/fn_size)


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
size_t get_serialized_size_px4_msgs__msg__RoverSpeedStatus(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RoverSpeedStatus__ros_msg_type * ros_message = static_cast<const _RoverSpeedStatus__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: timestamp
  {
    size_t item_size = sizeof(ros_message->timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: measured_speed_body_x
  {
    size_t item_size = sizeof(ros_message->measured_speed_body_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    size_t item_size = sizeof(ros_message->adjusted_speed_body_x_setpoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: pid_throttle_body_x_integral
  {
    size_t item_size = sizeof(ros_message->pid_throttle_body_x_integral);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: measured_speed_body_y
  {
    size_t item_size = sizeof(ros_message->measured_speed_body_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    size_t item_size = sizeof(ros_message->adjusted_speed_body_y_setpoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: pid_throttle_body_y_integral
  {
    size_t item_size = sizeof(ros_message->pid_throttle_body_y_integral);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}


ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
size_t max_serialized_size_px4_msgs__msg__RoverSpeedStatus(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;

  // Field name: timestamp
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: measured_speed_body_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: pid_throttle_body_x_integral
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: measured_speed_body_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: pid_throttle_body_y_integral
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }


  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = px4_msgs__msg__RoverSpeedStatus;
    is_plain =
      (
      offsetof(DataType, pid_throttle_body_y_integral) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
bool cdr_serialize_key_px4_msgs__msg__RoverSpeedStatus(
  const px4_msgs__msg__RoverSpeedStatus * ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  // Field name: timestamp
  {
    cdr << ros_message->timestamp;
  }

  // Field name: measured_speed_body_x
  {
    cdr << ros_message->measured_speed_body_x;
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    cdr << ros_message->adjusted_speed_body_x_setpoint;
  }

  // Field name: pid_throttle_body_x_integral
  {
    cdr << ros_message->pid_throttle_body_x_integral;
  }

  // Field name: measured_speed_body_y
  {
    cdr << ros_message->measured_speed_body_y;
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    cdr << ros_message->adjusted_speed_body_y_setpoint;
  }

  // Field name: pid_throttle_body_y_integral
  {
    cdr << ros_message->pid_throttle_body_y_integral;
  }

  return true;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
size_t get_serialized_size_key_px4_msgs__msg__RoverSpeedStatus(
  const void * untyped_ros_message,
  size_t current_alignment)
{
  const _RoverSpeedStatus__ros_msg_type * ros_message = static_cast<const _RoverSpeedStatus__ros_msg_type *>(untyped_ros_message);
  (void)ros_message;

  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  (void)padding;
  (void)wchar_size;

  // Field name: timestamp
  {
    size_t item_size = sizeof(ros_message->timestamp);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: measured_speed_body_x
  {
    size_t item_size = sizeof(ros_message->measured_speed_body_x);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    size_t item_size = sizeof(ros_message->adjusted_speed_body_x_setpoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: pid_throttle_body_x_integral
  {
    size_t item_size = sizeof(ros_message->pid_throttle_body_x_integral);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: measured_speed_body_y
  {
    size_t item_size = sizeof(ros_message->measured_speed_body_y);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    size_t item_size = sizeof(ros_message->adjusted_speed_body_y_setpoint);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  // Field name: pid_throttle_body_y_integral
  {
    size_t item_size = sizeof(ros_message->pid_throttle_body_y_integral);
    current_alignment += item_size +
      eprosima::fastcdr::Cdr::alignment(current_alignment, item_size);
  }

  return current_alignment - initial_alignment;
}

ROSIDL_TYPESUPPORT_FASTRTPS_C_PUBLIC_px4_msgs
size_t max_serialized_size_key_px4_msgs__msg__RoverSpeedStatus(
  bool & full_bounded,
  bool & is_plain,
  size_t current_alignment)
{
  size_t initial_alignment = current_alignment;

  const size_t padding = 4;
  const size_t wchar_size = 4;
  size_t last_member_size = 0;
  (void)last_member_size;
  (void)padding;
  (void)wchar_size;

  full_bounded = true;
  is_plain = true;
  // Field name: timestamp
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint64_t);
    current_alignment += array_size * sizeof(uint64_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint64_t));
  }

  // Field name: measured_speed_body_x
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: adjusted_speed_body_x_setpoint
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: pid_throttle_body_x_integral
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: measured_speed_body_y
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: adjusted_speed_body_y_setpoint
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  // Field name: pid_throttle_body_y_integral
  {
    size_t array_size = 1;
    last_member_size = array_size * sizeof(uint32_t);
    current_alignment += array_size * sizeof(uint32_t) +
      eprosima::fastcdr::Cdr::alignment(current_alignment, sizeof(uint32_t));
  }

  size_t ret_val = current_alignment - initial_alignment;
  if (is_plain) {
    // All members are plain, and type is not empty.
    // We still need to check that the in-memory alignment
    // is the same as the CDR mandated alignment.
    using DataType = px4_msgs__msg__RoverSpeedStatus;
    is_plain =
      (
      offsetof(DataType, pid_throttle_body_y_integral) +
      last_member_size
      ) == ret_val;
  }
  return ret_val;
}


static bool _RoverSpeedStatus__cdr_serialize(
  const void * untyped_ros_message,
  eprosima::fastcdr::Cdr & cdr)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  const px4_msgs__msg__RoverSpeedStatus * ros_message = static_cast<const px4_msgs__msg__RoverSpeedStatus *>(untyped_ros_message);
  (void)ros_message;
  return cdr_serialize_px4_msgs__msg__RoverSpeedStatus(ros_message, cdr);
}

static bool _RoverSpeedStatus__cdr_deserialize(
  eprosima::fastcdr::Cdr & cdr,
  void * untyped_ros_message)
{
  if (!untyped_ros_message) {
    fprintf(stderr, "ros message handle is null\n");
    return false;
  }
  px4_msgs__msg__RoverSpeedStatus * ros_message = static_cast<px4_msgs__msg__RoverSpeedStatus *>(untyped_ros_message);
  (void)ros_message;
  return cdr_deserialize_px4_msgs__msg__RoverSpeedStatus(cdr, ros_message);
}

static uint32_t _RoverSpeedStatus__get_serialized_size(const void * untyped_ros_message)
{
  return static_cast<uint32_t>(
    get_serialized_size_px4_msgs__msg__RoverSpeedStatus(
      untyped_ros_message, 0));
}

static size_t _RoverSpeedStatus__max_serialized_size(char & bounds_info)
{
  bool full_bounded;
  bool is_plain;
  size_t ret_val;

  ret_val = max_serialized_size_px4_msgs__msg__RoverSpeedStatus(
    full_bounded, is_plain, 0);

  bounds_info =
    is_plain ? ROSIDL_TYPESUPPORT_FASTRTPS_PLAIN_TYPE :
    full_bounded ? ROSIDL_TYPESUPPORT_FASTRTPS_BOUNDED_TYPE : ROSIDL_TYPESUPPORT_FASTRTPS_UNBOUNDED_TYPE;
  return ret_val;
}


static message_type_support_callbacks_t __callbacks_RoverSpeedStatus = {
  "px4_msgs::msg",
  "RoverSpeedStatus",
  _RoverSpeedStatus__cdr_serialize,
  _RoverSpeedStatus__cdr_deserialize,
  _RoverSpeedStatus__get_serialized_size,
  _RoverSpeedStatus__max_serialized_size,
  nullptr
};

static rosidl_message_type_support_t _RoverSpeedStatus__type_support = {
  rosidl_typesupport_fastrtps_c__identifier,
  &__callbacks_RoverSpeedStatus,
  get_message_typesupport_handle_function,
  &px4_msgs__msg__RoverSpeedStatus__get_type_hash,
  &px4_msgs__msg__RoverSpeedStatus__get_type_description,
  &px4_msgs__msg__RoverSpeedStatus__get_type_description_sources,
};

const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_fastrtps_c, px4_msgs, msg, RoverSpeedStatus)() {
  return &_RoverSpeedStatus__type_support;
}

#if defined(__cplusplus)
}
#endif
